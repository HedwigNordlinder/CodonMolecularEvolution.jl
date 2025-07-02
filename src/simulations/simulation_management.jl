function run_simulation_batch(config_file::String, output_dir::String; resume=false, generate_reports=true)
    mkpath(output_dir)
    config_df = CSV.read(config_file, DataFrame)
    log_file = joinpath(output_dir, "simulation_log.txt")
    
    # Calculate total number of simulations including replicates
    total_sims = sum(config_df.n_replicates)
    println("Starting batch simulation with $(nrow(config_df)) scenarios ($(total_sims) total simulations) using $(Threads.nthreads()) threads...")
    
    completed = Threads.Atomic{Int}(0)
    file_lock = ReentrantLock()
    
    # Create expanded task list for all replicates
    tasks = Tuple{Int, Int, String, String}[]  # (row_idx, replicate, scenario_name, full_name)
    for i in 1:nrow(config_df)
        row = config_df[i, :]
        scenario_name = String(row.scenario_name)
        n_reps = Int(row.n_replicates)
        
        for rep in 1:n_reps
            if n_reps == 1
                full_name = scenario_name
            else
                full_name = "$(scenario_name)_rep$(rep)"
            end
            push!(tasks, (i, rep, scenario_name, full_name))
        end
    end
    
    # Thread-safe storage for completed results
    completed_results = Vector{Union{SimulationResult, Nothing}}(nothing, length(tasks))
    completed_names = Vector{String}(undef, length(tasks))
    completion_flags = Vector{Bool}(undef, length(tasks))
    fill!(completion_flags, false)
    
    # Create progress bar
    progress = Progress(total_sims, desc="Simulations: ", showspeed=true)
    
    # Multithreaded simulation phase
    Threads.@threads for task_idx in 1:length(tasks)
        row_idx, replicate, scenario_name, full_name = tasks[task_idx]
        row = config_df[row_idx, :]
        scenario_dir = joinpath(output_dir, full_name)
        
        if resume && isdir(scenario_dir) && isfile(joinpath(scenario_dir, "$(full_name).nwk"))
            Threads.atomic_add!(completed, 1)
            next!(progress)  # Update progress bar
            continue
        end
        
        try
            params = parse_simulation_parameters(row)
            result = run_single_simulation(params...)
            
            # Thread-safe directory creation and file saving
            lock(file_lock) do
                mkpath(scenario_dir)
            end
            save_simulation_data(result; name=joinpath(scenario_dir, full_name))
            
            # Store result for report generation
            if generate_reports
                completed_results[task_idx] = result
                completed_names[task_idx] = full_name
                completion_flags[task_idx] = true
            end
            
            # Thread-safe logging
            log_message = "SUCCESS - $full_name (Thread $(Threads.threadid()))\n"
            lock(file_lock) do
                open(log_file, "a") do f
                    write(f, log_message)
                end
            end
            
        catch e
            error_msg = "ERROR - $full_name: $e (Thread $(Threads.threadid()))\n"
            lock(file_lock) do
                open(log_file, "a") do f
                    write(f, error_msg)
                end
            end
        end
        
        # Update progress bar (thread-safe)
        Threads.atomic_add!(completed, 1)
        next!(progress)
    end
    
    # Finish progress bar
    finish!(progress)
    println("Batch simulation complete. Check $log_file for details.")
    
    # Single-threaded report generation phase
    if generate_reports
        successful_indices = findall(completion_flags)
        if !isempty(successful_indices)
            println("Generating tree reports for $(length(successful_indices)) simulations (single-threaded)...")
            
            # Optional: Add progress bar for report generation too
            report_progress = Progress(length(successful_indices), desc="Reports: ")
            
            for i in successful_indices
                result = completed_results[i]
                name = completed_names[i]
                scenario_dir = joinpath(output_dir, name)
                report_filename = joinpath(scenario_dir, "$(name)_tree_report")
                
                try
                    save_tree_report(result, report_filename)
                catch e
                    println("Error generating report for $name: $e")
                end
                
                next!(report_progress)
            end
            
            finish!(report_progress)
            println("Report generation complete.")
        end
    end
end

function parse_simulation_parameters(row)
    ntaxa = row.ntaxa
    nsites = row.nsites
    
    # Parse scenario
    scenario = eval(Meta.parse(String(row.coalescence_scenario)))
    
    # Get matrices
    nucleotide_matrix = get_nucleotide_matrix(String(row.nucleotide_model))
    f3x4_matrix = get_f3x4_matrix(String(row.f3x4_model))
    
    # Create rate sampler
    rate_sampler = create_rate_sampler_from_row(row)
    
    target_normalisation = get(row, :target_normalisation, 1.0)
    
    return (ntaxa, scenario, rate_sampler, nsites, nucleotide_matrix, f3x4_matrix, target_normalisation)
end

function create_rate_sampler_from_row(row)
    # Allow direct specification of sampler in Julia code
    if hasproperty(row, :rate_sampler) && !ismissing(row.rate_sampler)
        return eval(Meta.parse(String(row.rate_sampler)))
    else
        error("CSV must contain 'rate_sampler' column with Julia code specifying the sampler")
    end
end

function run_single_simulation(ntaxa, scenario, rate_sampler::RateSampler, nsites, 
                              nucleotide_matrix, f3x4_matrix, target_normalisation)
    return simulate_alignment(ntaxa, scenario, rate_sampler, nsites, nucleotide_matrix, 
                            f3x4_matrix; target_normalisation=target_normalisation)
end

# Constant equal to demo_nucmat for now
function get_nucleotide_matrix(model_name::String)
    return CodonMolecularEvolution.demo_nucmat
end

function get_f3x4_matrix(model_name::String)
    return CodonMolecularEvolution.demo_f3x4
end

function save_tree_report(results::SimulationResult, output_filename = nothing) 
    println("Plots.jl and Phylo.jl not loaded, not saving anything")
end
"""
    CodonMolecularEvolution.collect_global_values(parent_directory::String; 
                                                 results_subfolder::String="results",
                                                 output_filename::String="global_values.csv",
                                                 verbosity::Int=1)

Collect global values from .global files across all subdirectories and save to CSV.

# Arguments
- `parent_directory::String`: Path to the parent directory containing subdirectories
- `results_subfolder::String="results"`: Name of subfolder containing .global files
- `output_filename::String="global_values.csv"`: Output CSV filename
- `verbosity::Int=1`: Control level of output messages

# Description
This function:
1. Scans the parent directory for subdirectories
2. For each subdirectory, looks in the results subfolder for .global files
3. Extracts the single float value from each .global file
4. Organizes data by method name (filename without .global extension)
5. Saves all data to a CSV with format: subfolder_name, method1_value, method2_value, ...

# Returns
- `DataFrame`: The collected data as a DataFrame
"""
function collect_global_values(parent_directory::String; 
                                                     results_subfolder::String="results",
                                                     output_filename::String="global_values.csv",
                                                     verbosity::Int=1)
    
    if !isdir(parent_directory)
        error("Parent directory does not exist: $parent_directory")
    end
    
    # Get all subdirectories
    subdirs = [d for d in readdir(parent_directory, join=true) if isdir(d)]
    
    if isempty(subdirs)
        error("No subdirectories found in $parent_directory")
    end
    
    if verbosity > 0
        println("Found $(length(subdirs)) subdirectories to process")
    end
    
    # Dictionary to store all data: method_name => [values...]
    method_data = Dict{String, Vector{Union{Float64, Missing}}}()
    subfolder_names = String[]
    
    # First pass: collect all method names and subfolder names
    all_method_names = Set{String}()
    
    for subdir in subdirs
        dirname = basename(subdir)
        results_dir = joinpath(subdir, results_subfolder)
        
        if !isdir(results_dir)
            if verbosity > 0
                println("  No results directory found in $dirname")
            end
            continue
        end
        
        # Find all .global files
        global_files = filter(f -> endswith(f, ".global"), readdir(results_dir))
        
        if !isempty(global_files)
            push!(subfolder_names, dirname)
            
            for global_file in global_files
                method_name = replace(global_file, ".global" => "")
                push!(all_method_names, method_name)
            end
        elseif verbosity > 0
            println("  No .global files found in $dirname")
        end
    end
    
    if isempty(subfolder_names)
        error("No valid subdirectories with .global files found")
    end
    
    # Sort method names for consistent column ordering
    sorted_method_names = sort(collect(all_method_names))
    
    if verbosity > 0
        println("Found methods: $(join(sorted_method_names, ", "))")
        println("Processing $(length(subfolder_names)) valid subdirectories")
    end
    
    # Initialize data structure
    for method_name in sorted_method_names
        method_data[method_name] = Vector{Union{Float64, Missing}}(missing, length(subfolder_names))
    end
    
    # Second pass: collect the actual values
    for (idx, subdir) in enumerate([joinpath(parent_directory, name) for name in subfolder_names])
        dirname = basename(subdir)
        results_dir = joinpath(subdir, results_subfolder)
        
        if verbosity > 0
            println("Processing: $dirname")
        end
        
        # Find all .global files
        global_files = filter(f -> endswith(f, ".global"), readdir(results_dir))
        
        for global_file in global_files
            method_name = replace(global_file, ".global" => "")
            global_file_path = joinpath(results_dir, global_file)
            
            try
                # Read the single float from the file
                content = strip(read(global_file_path, String))
                value = parse(Float64, content)
                
                method_data[method_name][idx] = value
                
                if verbosity > 1
                    println("  $method_name: $value")
                end
                
            catch e
                if verbosity > 0
                    println("  ✗ Failed to read $global_file: $(e)")
                end
                # Value remains missing
            end
        end
    end
    
    # Create DataFrame
    df_data = Dict{String, Any}()
    df_data["subfolder"] = subfolder_names
    
    for method_name in sorted_method_names
        df_data[method_name] = method_data[method_name]
    end
    
    df = DataFrame(df_data)
    
    # Save to CSV
    output_path = joinpath(parent_directory, output_filename)
    CSV.write(output_path, df)
    
    if verbosity > 0
        println("Results saved to: $output_path")
        println("CSV contains $(nrow(df)) rows and $(ncol(df)) columns")
        
        # Show summary of missing values
        for method_name in sorted_method_names
            n_missing = sum(ismissing.(method_data[method_name]))
            if n_missing > 0
                println("  $method_name: $n_missing missing values")
            end
        end
    end
    
    return df
end