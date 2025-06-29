"""
    run_fubar_benchmark(input_directory::String, methods::Vector{<:FUBARMethod}; 
                       results_subfolder::String="results", 
                       verbosity::Int=1,
                       fasta_extensions::Vector{String}=["fasta"],
                       tree_extensions::Vector{String}=["nwk"],
                       nthreads::Union{Int,Nothing}=nothing)

Run multiple FUBAR methods on all subdirectories containing alignment data.

# Arguments
- `input_directory::String`: Path to the directory containing subdirectories with alignment data
- `methods::Vector{<:FUBARMethod}`: Array of FUBAR methods to run (e.g., [DirichletFUBAR(), FIFEFUBAR(), SKBDIFUBAR()])

# Keywords
- `results_subfolder::String="results"`: Name of subfolder to store results in each directory
- `verbosity::Int=1`: Control level of output messages
- `fasta_extensions::Vector{String}=["fasta"]`: File extensions to look for alignment files
- `tree_extensions::Vector{String}=["nwk"]`: File extensions to look for tree files
- `nthreads::Union{Int,Nothing}=nothing`: Number of threads to use (default: all available)

# Description
This function:
1. Scans the input directory for subdirectories
2. For each subdirectory, looks for alignment files and corresponding tree files
3. Runs each specified FUBAR method on the data (parallelized across directories)
4. Saves results to CSV files in a 'results' subfolder
5. Disables plotting for all analyses

# Returns
- `Dict{String, Dict{String, Any}}`: Nested dictionary with results for each directory and method
"""
function run_fubar_benchmark(input_directory::String, methods::Vector{<:FUBARMethod}; 
                           results_subfolder::String="results", 
                           verbosity::Int=1,
                           fasta_extensions::Vector{String}=["fasta"],
                           tree_extensions::Vector{String}=["nwk"],
                           nthreads::Union{Int,Nothing}=nothing)
    
    if !isdir(input_directory)
        error("Input directory does not exist: $input_directory")
    end
    
    # Set number of threads
    if isnothing(nthreads)
        nthreads = Threads.nthreads()
    end
    
    if verbosity > 0
        println("Using $nthreads threads for parallel processing")
    end
    
    # Get all subdirectories
    subdirs = [d for d in readdir(input_directory, join=true) if isdir(d)]
    
    if isempty(subdirs)
        error("No subdirectories found in $input_directory")
    end
    
    if verbosity > 0
        println("Found $(length(subdirs)) subdirectories to process")
        println("Running $(length(methods)) FUBAR methods on each directory")
    end
    
    # Prepare work items for parallel processing
    work_items = []
    for subdir in subdirs
        dirname = basename(subdir)
        
        # Find alignment and tree files with specified extensions
        fasta_files = filter(f -> any(endswith(f, ".$ext") for ext in fasta_extensions), readdir(subdir, join=true))
        tre_files = filter(f -> any(endswith(f, ".$ext") for ext in tree_extensions), readdir(subdir, join=true))
        
        if isempty(fasta_files)
            if verbosity > 0
                println("  No alignment files found in $dirname with extensions: $(join(fasta_extensions, ", "))")
            end
            continue
        end
        
        # Use the first alignment file found
        fasta_file = fasta_files[1]
        tre_file = isempty(tre_files) ? nothing : tre_files[1]
        
        push!(work_items, (dirname, subdir, fasta_file, tre_file))
    end
    
    if isempty(work_items)
        error("No valid work items found")
    end
    
    # Process directories in parallel
    results = Dict{String, Dict{String, Any}}()
    
    if nthreads == 1 || length(work_items) == 1
        # Sequential processing for single thread or single work item
        for (dirname, subdir, fasta_file, tre_file) in work_items
            if verbosity > 0
                println("\nProcessing directory: $dirname")
            end
            
            dir_result = process_single_directory(dirname, subdir, fasta_file, tre_file, 
                                                methods, results_subfolder, verbosity)
            results[dirname] = dir_result
        end
    else
        # Parallel processing
        results_lock = ReentrantLock()
        
        Threads.@threads for (dirname, subdir, fasta_file, tre_file) in work_items
            if verbosity > 0
                lock(results_lock) do
                    println("\nProcessing directory: $dirname (Thread $(Threads.threadid()))")
                end
            end
            
            dir_result = process_single_directory(dirname, subdir, fasta_file, tre_file, 
                                                methods, results_subfolder, verbosity)
            
            lock(results_lock) do
                results[dirname] = dir_result
            end
        end
    end
    
    if verbosity > 0
        println("\nBenchmark completed!")
        println("Results saved in '$(results_subfolder)' subfolders")
    end
    
    return results
end

"""
    process_single_directory(dirname::String, subdir::String, fasta_file::String, tre_file::Union{String,Nothing}, 
                           methods::Vector{<:FUBARMethod}, results_subfolder::String, verbosity::Int)

Process a single directory with all FUBAR methods. This function is designed to be thread-safe.
"""
function process_single_directory(dirname::String, subdir::String, fasta_file::String, tre_file::Union{String,Nothing}, 
                                methods::Vector{<:FUBARMethod}, results_subfolder::String, verbosity::Int)
    
    try
        # Read alignment data
        seqs = read(fasta_file, FASTAReader)
        seqnames = [identifier(seq) for seq in seqs]
        sequences = [sequence(seq) for seq in seqs]
        
        # Read tree if available, otherwise use default
        if !isnothing(tre_file)
            treestring = read(tre_file, String)
        else
            # Create a simple star tree as fallback
            treestring = create_star_tree(seqnames)
        end
        
        # Create results subfolder
        results_dir = joinpath(subdir, results_subfolder)
        mkpath(results_dir)
        
        dir_results = Dict{String, Any}()
        
        # Run each FUBAR method
        for method in methods
            method_name = string(typeof(method).name.name)
            if verbosity > 0
                println("  Running $method_name...")
            end
            
            try
                # Create grid
                grid = alphabetagrid(seqnames, sequences, treestring, verbosity=verbosity-1)
                
                # Run analysis with plotting disabled using dispatch
                analysis_result = run_single_fubar_analysis(method, grid, results_dir, method_name, verbosity)
                dir_results[method_name] = analysis_result
                
                if verbosity > 0
                    println("    ✓ Completed successfully")
                end
                
            catch e
                if verbosity > 0
                    println("    ✗ Failed: $(e)")
                end
                dir_results[method_name] = (error=e,)
            end
        end
        
        return dir_results
        
    catch e
        if verbosity > 0
            println("  ✗ Failed to process directory: $(e)")
        end
        return Dict{String, Any}("error" => e)
    end
end

"""
    run_single_fubar_analysis(method::FUBARMethod, grid::FUBARGrid, results_dir::String, method_name::String, verbosity::Int)

Run a single FUBAR analysis using dispatch. This function dispatches on the method type to call the appropriate FUBAR_analysis function.
"""
function run_single_fubar_analysis(method::FUBARMethod, grid::FUBARGrid, results_dir::String, method_name::String, verbosity::Int)
    # This will dispatch to the appropriate FUBAR_analysis method based on the method type
    return FUBAR_analysis(method, grid, 
        analysis_name=joinpath(results_dir, lowercase(method_name)),
        exports=true, 
        verbosity=verbosity-1,
        disable_plotting=true)
end
