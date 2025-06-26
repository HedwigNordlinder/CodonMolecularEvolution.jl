function run_simulation_batch(config_file::String, output_dir::String; resume=false)
    # Create output directory if it doesn't exist
    mkpath(output_dir)
    
    # Read configuration CSV
    config_df = CSV.read(config_file, DataFrame)
    
    # Log file for tracking progress and errors
    log_file = joinpath(output_dir, "simulation_log.txt")
    
    println("Starting batch simulation with $(nrow(config_df)) scenarios using $(Threads.nthreads()) threads...")
    
    # Create thread-safe progress counter
    completed = Threads.Atomic{Int}(0)
    total = nrow(config_df)
    
    # Parallelize the main loop
    Threads.@threads for i in 1:nrow(config_df)
        row = config_df[i, :]
        scenario_name = row.scenario_name
        scenario_dir = joinpath(output_dir, scenario_name)
        
        # Skip if already completed and resume=true
        if resume && isdir(scenario_dir) && isfile(joinpath(scenario_dir, "$(scenario_name).nwk"))
            Threads.atomic_add!(completed, 1)
            println("Thread $(Threads.threadid()): Skipping completed scenario: $scenario_name")
            continue
        end
        
        println("Thread $(Threads.threadid()): Running scenario $scenario_name")
        
        try
            # Parse parameters from row
            params = parse_simulation_parameters(row)
            
            # Run simulation
            result = run_single_simulation(params...)
            
            # Save results (ensure thread-safe directory creation)
            lock = ReentrantLock()
            lock(lock) do
                mkpath(scenario_dir)
            end
            save_simulation_data(result; name=joinpath(scenario_dir, scenario_name))
            
            # Thread-safe logging
            log_message = "$(now()): SUCCESS - $scenario_name (Thread $(Threads.threadid()))\n"
            lock(lock) do
                open(log_file, "a") do f
                    write(f, log_message)
                end
            end
            
        catch e
            # Thread-safe error logging
            error_msg = "$(now()): ERROR - $scenario_name: $e (Thread $(Threads.threadid()))\n"
            println("Thread $(Threads.threadid()): Error in scenario $scenario_name: $e")
            lock = ReentrantLock()
            lock(lock) do
                open(log_file, "a") do f
                    write(f, error_msg)
                end
            end
        end
        
        # Update progress counter
        current_completed = Threads.atomic_add!(completed, 1) + 1
        if current_completed % 10 == 0 || current_completed == total
            println("Progress: $current_completed/$total completed")
        end
    end
    
    println("Batch simulation complete. Check $log_file for details.")
end

function parse_simulation_parameters(row)
    # Extract basic parameters
    ntaxa = row.ntaxa
    nsites = row.nsites
    
    # Parse distributions (assuming string format like "Gamma(2.0, 1.0)")
    alpha_dist = eval(Meta.parse(row.alpha_distribution))
    beta_dist = eval(Meta.parse(row.beta_distribution))
    
    # Parse scenario (assuming string format like "ConstantCoalescence(1000.0)")
    scenario = eval(Meta.parse(row.coalescence_scenario))
    
    # Get matrices (you might want these as parameters or constants)
    nucleotide_matrix = get_nucleotide_matrix(row.nucleotide_model)
    f3x4_matrix = get_f3x4_matrix(row.f3x4_model)
    
    # Optional parameters with defaults
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