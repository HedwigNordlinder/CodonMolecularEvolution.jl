module PlotsExt

using CodonMolecularEvolution
using Plots
using Measures
using MolecularEvolution
using Phylo
using DataFrames
using CSV
using Suppressor

const Dummy = CodonMolecularEvolution.PlotsExtDummy

# Interface between CodonMolecularEvolution and PlotsExt
#########################################################

function CodonMolecularEvolution.plot_tagged_phylo_tree(::Dummy, tree, tag_colors, tags, analysis_name; strip_tags_from_name=CodonMolecularEvolution.generate_tag_stripper(tags), exports=true)
    #TODO: update plots in docs
    phylo_tree = get_phylo_tree(tree)
    tagging = [tag_colors[CodonMolecularEvolution.model_ind(n, tags)] for n in nodenameiter(phylo_tree)]
    for node in nodeiter(phylo_tree)
        renamenode!(phylo_tree, node, strip_tags_from_name(node.name))
    end
    #Warnings regarding marker- and linecolor also appear in the Phylo.jl docs example
    #Note: sometimes long leafnames are truncated/not visible in the plot
    pl = plot(phylo_tree,
        showtips=true, tipfont=4, markercolor=tagging, linecolor=tagging, markerstrokewidth=0, size=(600, (120 + length(getleaflist(tree)) * 8)), margins = 1Plots.cm)
    exports && @suppress_err savefig_tweakSVG(analysis_name * "_tagged_input_tree.svg", pl)
    return pl
end

function CodonMolecularEvolution.FUBAR_plot_results(::Dummy, method::CodonMolecularEvolution.SKBDIFUBAR, results::CodonMolecularEvolution.BayesianFUBARResults, grid::CodonMolecularEvolution.FUBARGrid; analysis_name="skbdi_analysis", posterior_threshold = 0.95, volume_scaling = 1.0, exports=false, diagnostics=true)
    exports && plot_skbdi_mixing(results, grid, analysis_name)
    CodonMolecularEvolution.FUBAR_plot_results(Dummy(), CodonMolecularEvolution.DefaultBayesianFUBARMethod(), results, grid, analysis_name=analysis_name, posterior_threshold = posterior_threshold, volume_scaling = volume_scaling, exports=exports)
end
function CodonMolecularEvolution.FUBAR_plot_results(::Dummy, method::CodonMolecularEvolution.DirichletFUBAR, results::CodonMolecularEvolution.BayesianFUBARResults, grid::CodonMolecularEvolution.FUBARGrid; analysis_name="dirichlet_analysis", posterior_threshold = 0.95, volume_scaling = 1.0, exports=false, diagnostics=false)
    CodonMolecularEvolution.FUBAR_plot_results(Dummy(), CodonMolecularEvolution.DefaultBayesianFUBARMethod(), results, grid, analysis_name=analysis_name, posterior_threshold = posterior_threshold, volume_scaling = volume_scaling, exports=exports)
end

function CodonMolecularEvolution.FUBAR_plot_results(::Dummy, method::CodonMolecularEvolution.DefaultBayesianFUBARMethod, results::CodonMolecularEvolution.BayesianFUBARResults, grid::CodonMolecularEvolution.FUBARGrid; analysis_name="bayesian_analysis", posterior_threshold = 0.95, volume_scaling = 1.0, exports=false)
    posterior_mean_plot = gridplot(grid, results)
    positive_violin_plot, purifying_violin_plot = violin_plots(grid, results, posterior_threshold = posterior_threshold, volume_scaling = volume_scaling)
    if exports
        savefig(posterior_mean_plot, analysis_name * "_posterior_mean.pdf")
        if !isnothing(positive_violin_plot)
            savefig(positive_violin_plot, analysis_name * "_positive_violin.pdf")
        end
        if !isnothing(purifying_violin_plot)
            savefig(purifying_violin_plot, analysis_name * "_purifying_violin.pdf")
        end
    end
    return posterior_mean_plot, violin_plots
end

function CodonMolecularEvolution.difFUBAR_plot_results(::Dummy, analysis_name, pos_thresh, detections, param_means, num_sites, omegagrid, detected_sites, group1_volumes, group2_volumes, alpha_volumes; tag_colors=DIFFUBAR_TAG_COLORS, verbosity=1, exports=true, sites_to_plot=nothing, plot_collection=NamedTuple[])
    

    sites = [1:num_sites;]

    #Select the sites that will get plotted, in case you want to customize this.
    if isnothing(sites_to_plot)
        sites_to_plot = detected_sites
    end

    if length(sites_to_plot) == 0
        verbosity > 0 && println("No sites detected above threshold.")
    else
        verbosity > 0 && println("Plotting alpha and omega distributions. If exports = true, saved as " * analysis_name * "_violin_*.pdf")

        #Assumes alpha and omega grids are the same:
        grd = round.(omegagrid, digits=3)

        ysize2 = 250 + 50 * length(sites[sites_to_plot])
        ysize0 = 250 + 20 * length(sites[sites_to_plot])
        
        Plots.CURRENT_PLOT.nullableplot = nothing # PyPlots close()

        pl = plot()
        FUBAR_violin_plot(sites[sites_to_plot], alpha_volumes[sites_to_plot] .* 0.65, grd, tag="α", color="green", legend_ncol=1, y_label="", x_label="")
        plot!(size=(400, ysize2), grid=false, top_margin=15mm, left_margin=10mm, bottom_margin=10mm)

        exports && savefig(analysis_name * "_violin_alpha.pdf")
        push!(plot_collection, (;posterior_alpha = pl))

        Plots.CURRENT_PLOT.nullableplot = nothing # PyPlots close()

        #Plot the G1 and G2 omegas
        pl = plot()
        FUBAR_violin_plot(sites[sites_to_plot], group1_volumes[sites_to_plot] .* 0.65, grd, tag="ω1", color=tag_colors[1], legend_ncol=2, y_label="", x_label="")
        FUBAR_violin_plot(sites[sites_to_plot], group2_volumes[sites_to_plot] .* 0.65, grd, tag="ω2", color=tag_colors[2], legend_ncol=2, y_label="", x_label="")
        plot!(size=(400, ysize2), grid=false, top_margin=15mm, left_margin=10mm, bottom_margin=10mm)

        exports && savefig(analysis_name * "_violin_omegas.pdf")
        push!(plot_collection, (;posterior_omegas = pl))
        
        Plots.CURRENT_PLOT.nullableplot = nothing

        #Plot all three parameters, using the v_offset to separate the alphas from the omegas
        pl = plot()
        FUBAR_violin_plot(sites[sites_to_plot], group1_volumes[sites_to_plot] .* 0.35, grd, tag="ω1", color=tag_colors[1], v_offset=-0.1, y_label="", x_label="")
        FUBAR_violin_plot(sites[sites_to_plot], group2_volumes[sites_to_plot] .* 0.35, grd, tag="ω2", color=tag_colors[2], v_offset=-0.1, y_label="", x_label="")
        FUBAR_violin_plot(sites[sites_to_plot], alpha_volumes[sites_to_plot] .* 0.35, grd, tag="α", color="green", v_offset=0.1, y_label="", x_label="")
        plot!(size=(400, ysize2), grid=false, top_margin=20mm, left_margin=10mm, bottom_margin=10mm)

        exports && savefig(analysis_name * "_violin_all_params.pdf")
        push!(plot_collection, (;posterior_alpha_and_omegas = pl))
        Plots.CURRENT_PLOT.nullableplot = nothing

        #Coerce the violin plot function to also viz the "detection" posteriors.
        floored_detec = [clamp.((d .- pos_thresh) .* 20, 0.0, 1.0) for d in detections[sites_to_plot]]
        pl = plot()
        FUBAR_violin_plot(sites[sites_to_plot], [[f[1], 0.0, 0.0, 0.0] for f in floored_detec] .* 0.45,
            ["P(ω1>ω2)", "P(ω2>ω1)", "P(ω1>1)", "P(ω2>1)"], tag="P(ω1>ω2)", color=tag_colors[1],
            vertical_ind=nothing, plot_legend=false, x_label = "", y_label="")
        FUBAR_violin_plot(sites[sites_to_plot], [[0.0, f[2], 0.0, 0.0] for f in floored_detec] .* 0.45,
            ["P(ω1>ω2)", "P(ω2>ω1)", "P(ω1>1)", "P(ω2>1)"], tag="P(ω2>ω1)", color=tag_colors[2],
            vertical_ind=nothing, plot_legend=false, x_label = "", y_label="")
        FUBAR_violin_plot(sites[sites_to_plot], [[0.0, 0.0, f[3], 0.0] for f in floored_detec] .* 0.45,
            ["P(ω1>ω2)", "P(ω2>ω1)", "P(ω1>1)", "P(ω2>1)"], tag="P(ω1>1)", color=tag_colors[1],
            vertical_ind=nothing, plot_legend=false, x_label = "", y_label="")
        FUBAR_violin_plot(sites[sites_to_plot], [[0.0, 0.0, 0.0, f[4]] for f in floored_detec] .* 0.45,
            ["P(ω1>ω2)", "P(ω2>ω1)", "P(ω1>1)", "P(ω2>1)"], tag="P(ω2>1)", color=tag_colors[2],
            vertical_ind=nothing, legend_ncol=2, x_label="", plot_legend=false, y_label="")

        lmargin_detect = 8 + length(sites_to_plot) / 8

        plot!(size=(250, ysize0), margins=1Plots.cm, legend=false, grid=false,
            ytickfont=18, bottom_margin=15mm, xtickfont=18)
        

        exports && savefig(analysis_name * "_detections.pdf")
        push!(plot_collection, (;detections = pl))
        Plots.CURRENT_PLOT.nullableplot = nothing

    end

    Plots.CURRENT_PLOT.nullableplot = nothing
    pl = plot()
    FUBAR_omega_plot(param_means, tag_colors, pos_thresh, detections, num_sites)
    xsize = 250 + 1.4 * length(sites)
    plot!(size=(xsize, 300), margins=1.2Plots.cm, grid=false, legendfontsize=6)
    push!(plot_collection, (;overview = pl))
    exports && savefig(analysis_name * "_site_omega_means.pdf")
    
end

# End of interface
#########################################################

# Moving on to PlotsExt functions

function gridplot(grid::CodonMolecularEvolution.FUBARGrid, results::CodonMolecularEvolution.BayesianFUBARResults; title="")
    θ = results.posterior_mean
    p = scatter(grid.alpha_ind_vec, grid.beta_ind_vec, zcolor=θ, c=:darktest, colorbar=false,
        markersize=sqrt(length(grid.alpha_ind_vec)) / 3.5, markershape=:square, markerstrokewidth=0.0, size=(350, 350),
        label=:none, xticks=(1:length(grid.grid_values), round.(grid.grid_values, digits=3)), xrotation=90,
        yticks=(1:length(grid.grid_values), round.(grid.grid_values, digits=3)), margin=6Plots.mm,
        xlabel="α", ylabel="β", title=title)
    plot(p, 1:length(grid.grid_values), 1:length(grid.grid_values), color="grey", style=:dash, label=:none)
    return p
end

function FUBAR_violin_plot(sites, group1_volumes, omegagrid;
    color="black", tag="", alpha=0.6,
    x_label="Parameter", y_label="Codon Sites",
    v_offset=0.0, legend_ncol=3,
    vertical_ind=findfirst(omegagrid .>= 1.0),
    plot_legend=true)

    ypos = [-i * 0.5 for i in 1:length(sites)]
    if !isnothing(vertical_ind)
        bar!([vertical_ind], [2 + maximum(ypos) - minimum(ypos)], bottom=[minimum(ypos) - 1], color="grey", alpha=0.05, label=:none)
    end

    for i in 1:length(sites)
        center_line = ypos[i]
        a = 1:length(omegagrid)
        b = group1_volumes[i]
        c = (v_offset .+ center_line .+ (-0.5 .* group1_volumes[i]))

        bar!(a, b + c, fillto=c, linewidth=0, bar_edges=false, linealpha=0.0, ylims=(minimum(c) - 1, 0), color=color, alpha=alpha, label=:none)
    end

    bar!([-10], [1], bottom=[1000], color=color, alpha=alpha, label=tag, linewidth=0, bar_edges=false, linealpha=0.0)
    bar!(yticks=(ypos, ["       " for _ in sites]), xticks=((1:length(omegagrid)), ["       " for _ in 1:length(omegagrid)]), xrotation=90)
    annotate!(length(omegagrid) / 2, -length(sites) / 2 - (2.0 + (length(sites) / 500)), Plots.text(x_label, "black", :center, 10))
    annotate!(-4.5, -(length(sites) + 1) / 4, Plots.text(y_label, "black", :center, 10, rotation=90))

    bar!(ylim=(minimum(ypos) - 0.5, maximum(ypos) + 0.5), xlim=(0, length(omegagrid) + 1))
    if plot_legend
        plot!(
            legend=([0.5, 0.45, 0.4, 0.35, 0.3][legend_ncol], 1 + 0.5 / (50 + length(sites))),
            legendcolumns=legend_ncol,
            shadow=true, fancybox=true,
        )
    end
    for i in 1:length(sites)
        annotate!(-0.5, ypos[i], Plots.text("$(sites[i])", "black", :right, 9))
    end
    for i in 1:length(omegagrid)
        annotate!(i, -length(sites) / 2 - 0.55 - length(sites) / 3000, Plots.text("$(omegagrid[i])", "black", :right, 9, rotation=90))
    end
end

function FUBAR_omega_plot(param_means, tag_colors, pos_thresh, detections, num_sites)
    #A plot of the omega means for all sites.
    omega1_means = [p[2] for p in param_means]
    omega2_means = [p[3] for p in param_means]

    t(x) = log10(x + 1)
    invt(y) = 10^y - 1

    omega1_means = t.(omega1_means)
    omega2_means = t.(omega2_means)

    for i in 1:length(omega1_means)
        tc = "black"
        if omega1_means[i] > omega2_means[i]
            tc = tag_colors[1]
        else
            tc = tag_colors[2]
        end

        diff_mul = 1.0
        if !(maximum(detections[i][1:2]) > pos_thresh)
            diff_mul = 0.1
        end

        pos1_mul = 1.0
        if !(detections[i][3] > pos_thresh)
            pos1_mul = 0.15
        end

        pos2_mul = 1.0
        if !(detections[i][4] > pos_thresh)
            pos2_mul = 0.15
        end
        plot!([i, i], [omega1_means[i], omega2_means[i]], color=tc, alpha=0.75 * diff_mul, linewidth=2, xlim=(-4, num_sites + 5), label="", yscale=:log10)
        scatter!([i], [omega1_means[i]], color=tag_colors[1], alpha=0.75 * pos1_mul, ms=2.5, label="", markerstrokecolor=:auto, yscale=:log10)
        scatter!([i], [omega2_means[i]], color=tag_colors[2], alpha=0.75 * pos2_mul, ms=2.5, label="", markerstrokecolor=:auto, yscale=:log10)
    end


    scatter!([-100, -100], [2, 2], color=tag_colors[1], alpha=0.75, label="ω1>1", ms=2.5)
    scatter!([-100, -100], [2, 2], color=tag_colors[2], alpha=0.75, label="ω2>1", ms=2.5)
    plot!([-100, -100], [2, 2], color=tag_colors[1], alpha=0.75, label="ω1>ω2", linewidth=2)
    plot!([-100, -100], [2, 2], color=tag_colors[2], alpha=0.75, label="ω2>ω1", linewidth=2)
    plot!([-2, num_sites + 3], [log10(1.0 + 1), log10(1.0 + 1)], color="grey", alpha=0.5, label="ω=1", linestyle=:dot, linewidth=2)
    xlabel!("Codon Sites")
    ylabel!("ω")

    n_points = 8
    lb = 0.01
    ub = 10
    points = collect(t(lb):(t(ub)-t(lb))/(n_points-1):t(ub))
    ticklabels = string.(round.(invt.(points), sigdigits=2))
    yticks!(points, ticklabels)

    xticks!(0:50:num_sites)

    plot!(
        legend=:outertop,
        legendcolumns=5,
        ylim=(0, log10(11)))

end

function violin_plots(grid::CodonMolecularEvolution.FUBARGrid, results::CodonMolecularEvolution.BayesianFUBARResults;
    posterior_threshold=0.95,
    volume_scaling=1.0,
    plot_positive=true,
    plot_purifying=true)

    # Extract grid values for the x-axis
    grid_values = grid.grid_values
    grd = round.(grid_values, digits=3)

    # Find sites with significant positive selection
    sites_positive = findall(results.positive_posteriors .> posterior_threshold)
    num_positive = length(sites_positive)
#    println("$(num_positive) sites with positive selection above threshold.")

    p_positive = nothing
    p_purifying = nothing

    if num_positive > 0 && plot_positive
        p_positive = plot()
        # Calculate scaling factor for distributions
        s = 0.5 / max(maximum(results.posterior_alpha[:, sites_positive]),
            maximum(results.posterior_beta[:, sites_positive]))

        # Plot alpha distributions for positively selected sites
        FUBAR_violin_plot(sites_positive,
            [s .* volume_scaling .* results.posterior_alpha[:, [i]] for i in sites_positive],
            grd, tag="α", color="blue", legend_ncol=2, vertical_ind=nothing)

        # Plot beta distributions for positively selected sites
        FUBAR_violin_plot(sites_positive,
            [s .* volume_scaling .* results.posterior_beta[:, [i]] for i in sites_positive],
            grd, tag="β", color="red", legend_ncol=2, vertical_ind=nothing)

        plot!(size=(400, num_positive * 17 + 300), grid=false, margin=15Plots.mm)
    end

    # Find sites with significant purifying selection
    sites_purifying = findall(results.purifying_posteriors .> posterior_threshold)
    num_purifying = length(sites_purifying)
    #println("$(num_purifying) sites with purifying selection above threshold.")

    if num_purifying > 0 && plot_purifying
        p_purifying = plot()
        # Calculate scaling factor for distributions
        s = 0.5 / max(maximum(results.posterior_alpha[:, sites_purifying]),
            maximum(results.posterior_beta[:, sites_purifying]))

        # Plot alpha distributions for purifying selected sites
        FUBAR_violin_plot(sites_purifying,
            [s .* volume_scaling .* results.posterior_alpha[:, [i]] for i in sites_purifying],
            grd, tag="α", color="blue", legend_ncol=2, vertical_ind=nothing)

        # Plot beta distributions for purifying selected sites
        FUBAR_violin_plot(sites_purifying,
            [s .* volume_scaling .* results.posterior_beta[:, [i]] for i in sites_purifying],
            grd, tag="β", color="red", legend_ncol=2, vertical_ind=nothing)

        plot!(size=(400, num_purifying * 17 + 300), grid=false, margin=15Plots.mm)
    end

    return p_positive, p_purifying
end



function plot_skbdi_mixing(results::CodonMolecularEvolution.BayesianFUBARResults,
    grid::CodonMolecularEvolution.FUBARGrid,
    analysis_name::String)


    kernel_params = results.kernel_parameters

    if isempty(kernel_params)
        println("No kernel parameters found.")
        return nothing
    end

    num_samples = length(kernel_params)
    num_dims = length(kernel_params[1])
    println("Kernel parameter dimension: ",num_dims)

    param_matrix = zeros(num_samples, num_dims)
    for i in 1:num_samples
        for j in 1:num_dims
            param_matrix[i, j] = kernel_params[i][j]
        end
    end


    
     p = plot(layout=(2, num_dims), size=(300 * num_dims, 600),
        legend=false, title="Kernel Bandwidth Parameters - $analysis_name")


    for j in 1:num_dims
        plot!(p[j], 1:num_samples, param_matrix[:, j],
            title="Dimension $j Trace",
            xlabel="Sample", ylabel="Value",
            linewidth=1, color=j)
        histogram!(p[j+num_dims], param_matrix[:, j],
            title="Dimension $j Posterior",
            xlabel="Value", ylabel="Frequency",
            bins=round(Int, sqrt(num_samples)), 
            color=j, alpha=0.7)
    end
    savefig(p, "$(analysis_name)_kernel_bandwidth_mixing.pdf")
    return p
    
end

function plot_simulated_tree(result::SimulationResult)
    return plot(get_phylo_tree(result.tree))
end
function plot_loglog_rates(result::SimulationResult)

    # Create scatter plot with log-transformed data
    p1 = scatter(log.(result.alphavec), log.(result.betavec),
                xlabel = "log(α)",
                ylabel = "log(β)", 
                title = "Log-Log Plot of Rates: "*simulation_name,
                label = "Data points",
                alpha = 0.7,
                markersize = 4)
    
    # Add y = x reference line
    x_range = xlims(p1)
    plot!(p1, [x_range[1], x_range[2]], [x_range[1], x_range[2]], 
          line = :dash, color = :red, linewidth = 2, label = "y = x")
    
    # Create histogram for alpha
    p2 = histogram(result.alphavec,
                   xlabel = "α",
                   ylabel = "Frequency",
                   title = "Distribution of α: "*simulation_name,
                   alpha = 0.7,
                   color = :blue,
                   bins = 30)
    
    # Create histogram for beta
    p3 = histogram(result.betavec,
                   xlabel = "β",
                   ylabel = "Frequency", 
                   title = "Distribution of β: "*simulation_name,
                   alpha = 0.7,
                   color = :green,
                   bins = 30)
    
    return p1, p2, p3
end
function plot_scenario_information(scenario::Nothing) return nothing,nothing end
function plot_scenario_information(scenario::CoalescenceScenario) 
    Ne = t -> effective_population_size(scenario, t)
    s = t -> sampling_rate(scenario, t)  # renamed to avoid conflict
    
    # Create time vector from 0 to 1
    t_vec = range(0, 100, length=1000)
    
    # Evaluate functions over time
    Ne_values = [Ne(t) for t in t_vec]
    sampling_values = [s(t) for t in t_vec]
    
    # Create first plot - Effective Population Size
    p1 = plot(t_vec, Ne_values,
              xlabel="Time",
              ylabel="Effective Population Size (Ne)",
              title="Effective Population Size Over Time",
              linewidth=2,
              color=:blue,
              grid=true)
    
    # Create second plot - Sampling Rate
    p2 = plot(t_vec, sampling_values,
              xlabel="Time", 
              ylabel="Sampling Rate",
              title="Sampling Rate Over Time",
              linewidth=2,
              color=:red,
              grid=true)
    
    return p1, p2
end
function CodonMolecularEvolution.save_tree_report(result::CodonMolecularEvolution.SimulationResult, output_filename=nothing)
    # Set default filename if not provided
    if output_filename === nothing
        output_filename = simulation_name * "_tree_report.pdf"
    end
    
    # Generate all plots
    tree_plot = plot_tree(simulation_name)
    scatter_plot, alpha_hist, beta_hist = plot_loglog_rates(simulation_name)
    scenario_p1, scenario_p2 = plot_scenario_information(simulation_name)
    
    # Create a combined layout with all plots
    # Tree on top row
    # Three rate plots in middle row 
    # Two scenario plots in bottom row
    combined_plot = plot()
    if !isnothing(scenario_p1) 
        combined_plot = plot(tree_plot, 
                        scatter_plot, alpha_hist, beta_hist,
                        scenario_p1, scenario_p2,
                        layout = @layout([a; [b c d]; [e f]]),
                        size = (1200, 1200),
                        plot_title = "Tree Report: " * simulation_name,
                        plot_titlefontsize = 16)
    else 
        combined_plot = plot(tree_plot, 
        scatter_plot, alpha_hist, beta_hist,
        layout = @layout([a; [b c d]; [e f]]),
        size = (1200, 1200),
        plot_title = "Tree Report: " * simulation_name,
        plot_titlefontsize = 16)
    end
    savefig(combined_plot, output_filename)
    
    println("Tree report saved to: ", output_filename)
    return output_filename
end

end