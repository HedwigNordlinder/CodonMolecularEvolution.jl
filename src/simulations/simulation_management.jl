function run_simulation_batch(config_file::String, output_dir::String; resume=false, generate_reports=true)
    mkpath(output_dir)
    config_df = CSV.read(config_file, DataFrame)
    log_file = joinpath(output_dir, "simulation_log.txt")
    
    println("Starting batch simulation with $(nrow(config_df)) scenarios using $(Threads.nthreads()) threads...")
    
    completed = Threads.Atomic{Int}(0)
    total = nrow(config_df)
    
    # Create a single lock for all file operations
    file_lock = ReentrantLock()
    
    # Thread-safe storage for completed results
    completed_results = Vector{SimulationResult}(undef, nrow(config_df))
    completed_names = Vector{String}(undef, nrow(config_df))
    completion_flags = Vector{Bool}(undef, nrow(config_df))
    fill!(completion_flags, false)
    
    # Multithreaded simulation phase
    Threads.@threads for i in 1:nrow(config_df)
        row = config_df[i, :]
        scenario_name = String(row.scenario_name)
        scenario_dir = joinpath(output_dir, scenario_name)
        
        if resume && isdir(scenario_dir) && isfile(joinpath(scenario_dir, "$(scenario_name).nwk"))
            Threads.atomic_add!(completed, 1)
            println("Thread $(Threads.threadid()): Skipping completed scenario: $scenario_name")
            # Don't set completion_flags[i] = true for skipped scenarios if we want to exclude them from reports
            continue
        end
        
        println("Thread $(Threads.threadid()): Running scenario $scenario_name")
        
        try
            params = parse_simulation_parameters(row)
            result = run_single_simulation(params...)
            
            # Thread-safe directory creation and file saving
            lock(file_lock) do
                mkpath(scenario_dir)
            end
            save_simulation_data(result; name=joinpath(scenario_dir, scenario_name))
            
            # Store result for report generation (thread-safe since each thread writes to unique index)
            if generate_reports
                completed_results[i] = result
                completed_names[i] = scenario_name
                completion_flags[i] = true
            end
            
            # Thread-safe logging
            log_message = "SUCCESS - $scenario_name (Thread $(Threads.threadid()))\n"
            lock(file_lock) do
                open(log_file, "a") do f
                    write(f, log_message)
                end
            end
            
        catch e
            error_msg = "ERROR - $scenario_name: $e (Thread $(Threads.threadid()))\n"
            println("Thread $(Threads.threadid()): Error in scenario $scenario_name: $e")
            lock(file_lock) do
                open(log_file, "a") do f
                    write(f, error_msg)
                end
            end
        end
        
        current_completed = Threads.atomic_add!(completed, 1) + 1
        if current_completed % 10 == 0 || current_completed == total
            println("Progress: $current_completed/$total completed")
        end
    end
    
    println("Batch simulation complete. Check $log_file for details.")
    
    # Single-threaded report generation phase
    if generate_reports
        successful_indices = findall(completion_flags)
        if !isempty(successful_indices)
            println("Generating tree reports for $(length(successful_indices)) scenarios (single-threaded)...")
            for i in successful_indices
                result = completed_results[i]
                name = completed_names[i]
                scenario_dir = joinpath(output_dir, name)
                report_filename = joinpath(scenario_dir, "$(name)_tree_report")
                
                try
                    save_tree_report(result, report_filename)
                    println("Generated report for: $name")
                catch e
                    println("Error generating report for $name: $e")
                end
            end
            println("Report generation complete.")
        end
    end
end

function parse_simulation_parameters(row)
    ntaxa = row.ntaxa
    nsites = row.nsites
    
    # Convert string types to String and parse
    alpha_dist = eval(Meta.parse(String(row.alpha_distribution)))
    beta_dist = eval(Meta.parse(String(row.beta_distribution)))
    scenario = eval(Meta.parse(String(row.coalescence_scenario)))
    
    # Convert to String before passing to get functions
    nucleotide_matrix = get_nucleotide_matrix(String(row.nucleotide_model))
    f3x4_matrix = get_f3x4_matrix(String(row.f3x4_model))
    
    diversifying_sites = get(row, :diversifying_sites, 0)
    target_normalisation = get(row, :target_normalisation, 1.0)
    
    return (ntaxa, scenario, alpha_dist, beta_dist, nsites, diversifying_sites, 
            nucleotide_matrix, f3x4_matrix, target_normalisation)
end



function run_single_simulation(ntaxa, scenario, alpha_dist, beta_dist, nsites, 
                              diversifying_sites, nucleotide_matrix, f3x4_matrix, 
                              target_normalisation)
    
    if diversifying_sites > 0
        return simulate_k_diversifying_sites(ntaxa, scenario, alpha_dist, beta_dist, 
                                           nsites, diversifying_sites, nucleotide_matrix, 
                                           f3x4_matrix; target_normalisation=target_normalisation)
    else
        return simulate_alignment(ntaxa, scenario, alpha_dist, beta_dist, nsites, 
                                nucleotide_matrix, f3x4_matrix; 
                                target_normalisation=target_normalisation)
    end
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