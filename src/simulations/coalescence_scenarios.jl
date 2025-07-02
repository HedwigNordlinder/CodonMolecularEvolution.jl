abstract type CoalescenceScenario end

# Structs with all parameters
struct HIVScenario <: CoalescenceScenario
    baseline_Ne::Int
    treatment_epochs::Vector{Float64}
    bottleneck_severity::Float64
    resistance_recovery::Float64
    recovery_rate::Float64
    base_rate::Float64
    treatment_failure_boost::Float64
    failure_detection_delay::Float64
    sampling_window::Float64
end

struct InfluenzaScenario <: CoalescenceScenario
    baseline_Ne::Int
    antigenic_epochs::Vector{Float64}
    sweep_duration::Float64
    sweep_bottleneck::Float64
    seasonal_amplitude::Float64
    peak_time::Float64
    period::Float64
    base_rate::Float64
    geographic_bias::Float64
    nh_peak_time::Float64
    sh_peak_time::Float64
    surveillance_ramp_up::Vector{Float64}
    enhancement_factor::Float64
end

struct COVIDScenario <: CoalescenceScenario
    baseline_Ne::Int
    variant_emergences::Vector{Float64}
    growth_advantages::Vector{Float64}
    displacement_rate::Float64
    global_control_measures::Vector{Float64}
    base_rate::Float64
    sequencing_ramp_up::Float64
    variant_concern_periods::Vector{Float64}
    geographic_inequality::Float64
    concern_duration::Float64
    max_sampling_rate::Float64
end

# Constructors with sensible defaults
HIVScenario(;
    baseline_Ne=500,
    treatment_epochs=[0.3, 0.7],
    bottleneck_severity=0.1,
    resistance_recovery=2.0,
    recovery_rate=1.5,
    base_rate=0.02,
    treatment_failure_boost=15.0,
    failure_detection_delay=0.1,
    sampling_window=0.2
) = HIVScenario(baseline_Ne, treatment_epochs, bottleneck_severity,
    resistance_recovery, recovery_rate, base_rate,
    treatment_failure_boost, failure_detection_delay, sampling_window)

InfluenzaScenario(;
    baseline_Ne=2000,
    antigenic_epochs=[0.25, 0.6],
    sweep_duration=0.15,
    sweep_bottleneck=0.2,
    seasonal_amplitude=0.6,
    peak_time=0.25,
    period=1.0,
    base_rate=0.15,
    geographic_bias=2.0,
    nh_peak_time=0.25,
    sh_peak_time=0.75,
    surveillance_ramp_up=[0.1, 0.4],
    enhancement_factor=3.0
) = InfluenzaScenario(baseline_Ne, antigenic_epochs, sweep_duration, sweep_bottleneck,
    seasonal_amplitude, peak_time, period, base_rate, geographic_bias,
    nh_peak_time, sh_peak_time, surveillance_ramp_up, enhancement_factor)

COVIDScenario(;
    baseline_Ne=5000,
    variant_emergences=[0.15, 0.4, 0.65],
    growth_advantages=[1.5, 2.0, 2.5],
    displacement_rate=4.0,
    global_control_measures=[0.1, 0.8],
    base_rate=0.05,
    sequencing_ramp_up=0.2,
    variant_concern_periods=[0.15, 0.4, 0.65],
    geographic_inequality=3.0,
    concern_duration=0.1,
    max_sampling_rate=2.0
) = COVIDScenario(baseline_Ne, variant_emergences, growth_advantages, displacement_rate,
    global_control_measures, base_rate, sequencing_ramp_up,
    variant_concern_periods, geographic_inequality, concern_duration, max_sampling_rate)

# Generic dispatch functions
function effective_population_size(scenario::HIVScenario, t::Real)
    Ne = scenario.baseline_Ne

    for treatment_start in scenario.treatment_epochs
        if t >= treatment_start
            time_since_treatment = t - treatment_start
            bottleneck_factor = scenario.bottleneck_severity * exp(-scenario.recovery_rate * time_since_treatment) +
                                scenario.resistance_recovery * (1 - exp(-scenario.recovery_rate * time_since_treatment))
            Ne *= bottleneck_factor
        end
    end

    return max(10, Ne)
end

function effective_population_size(scenario::InfluenzaScenario, t::Real)
    seasonal_factor = 1 + scenario.seasonal_amplitude * cos(2π * (t / scenario.period - scenario.peak_time))
    Ne = scenario.baseline_Ne * seasonal_factor

    for epoch_start in scenario.antigenic_epochs
        if t >= epoch_start
            time_since_epoch = t - epoch_start
            if time_since_epoch <= scenario.sweep_duration
                sweep_progress = time_since_epoch / scenario.sweep_duration
                bottleneck_factor = 1 - (1 - scenario.sweep_bottleneck) * sin(π * sweep_progress)
                Ne *= bottleneck_factor
            end
        end
    end

    return max(100, Ne)
end

function effective_population_size(scenario::COVIDScenario, t::Real)
    Ne = scenario.baseline_Ne

    # Control measures
    for control_start in scenario.global_control_measures
        control_end = control_start + 0.15
        if control_start <= t <= control_end
            Ne *= 0.3
        end
    end

    # Variant sweeps
    for (i, emergence_time) in enumerate(scenario.variant_emergences)
        if t >= emergence_time
            time_since_emergence = t - emergence_time
            advantage = scenario.growth_advantages[i]
            sweep_progress = 1 / (1 + exp(-scenario.displacement_rate * (time_since_emergence - 0.1)))
            Ne *= (1 + (advantage - 1) * sweep_progress)
        end
    end

    return max(500, Ne)
end

function sampling_rate(scenario::HIVScenario, t::Real)
    rate = scenario.base_rate

    for treatment_start in scenario.treatment_epochs
        failure_window_start = treatment_start + scenario.failure_detection_delay
        failure_window_end = failure_window_start + scenario.sampling_window

        if failure_window_start <= t <= failure_window_end
            rate *= scenario.treatment_failure_boost
        end
    end

    return rate
end

function sampling_rate(scenario::InfluenzaScenario, t::Real)
    nh_factor = 1 + scenario.geographic_bias * exp(cos(2π * (t / scenario.period - scenario.nh_peak_time)))
    sh_factor = 1 + 0.3 * exp(cos(2π * (t / scenario.period - scenario.sh_peak_time)))

    rate = scenario.base_rate * (nh_factor + sh_factor) / 2

    for ramp_start in scenario.surveillance_ramp_up
        if ramp_start <= t <= ramp_start + 0.2
            rate *= scenario.enhancement_factor
        end
    end

    return rate
end

function sampling_rate(scenario::COVIDScenario, t::Real)
    if t >= scenario.sequencing_ramp_up
        capacity_factor = min(5.0, 1 + 4 * (t - scenario.sequencing_ramp_up) / 0.3)
    else
        capacity_factor = 0.2
    end

    rate = scenario.base_rate * capacity_factor * scenario.geographic_inequality

    for concern_start in scenario.variant_concern_periods
        if concern_start <= t <= concern_start + scenario.concern_duration
            rate = min(scenario.max_sampling_rate, rate * 4.0)
        end
    end

    return rate
end
struct LogisticGrowthScenario <: CoalescenceScenario
    carrying_capacity::Int          # K in logistic equation
    growth_rate::Float64           # r in logistic equation  
    initial_population::Int        # N₀
    inflection_time::Float64       # when growth rate peaks
    base_rate::Float64            # baseline sampling rate
    detection_threshold::Float64   # population size when outbreak detected
    detection_delay::Float64      # time lag in detection
    reactive_multiplier::Float64  # sampling intensification after detection
    investigation_duration::Float64 # how long intense sampling lasts
end

# Constructor with sensible defaults
LogisticGrowthScenario(;
    carrying_capacity=2000,
    growth_rate=8.0,               # relatively fast growth
    initial_population=10,
    inflection_time=0.4,           # midpoint of simulation timeframe
    base_rate=0.01,               # very low baseline sampling
    detection_threshold=100.0,     # outbreak detected at ~5% of K
    detection_delay=0.05,         # ~2 week detection lag
    reactive_multiplier=20.0,     # 20x sampling increase during investigation
    investigation_duration=0.3    # investigation lasts ~3 months
) = LogisticGrowthScenario(carrying_capacity, growth_rate, initial_population,
    inflection_time, base_rate, detection_threshold,
    detection_delay, reactive_multiplier, investigation_duration)

function effective_population_size(scenario::LogisticGrowthScenario, t::Real)
    # Logistic growth: N(t) = K / (1 + ((K - N₀)/N₀) * exp(-r * (t - t₀)))
    # Rearranged for numerical stability
    t_shifted = t - scenario.inflection_time
    exp_term = exp(-scenario.growth_rate * t_shifted)

    K = scenario.carrying_capacity
    N0 = scenario.initial_population

    Ne = K / (1 + ((K - N0) / N0) * exp_term)

    return max(scenario.initial_population, round(Int, Ne))
end

function sampling_rate(scenario::LogisticGrowthScenario, t::Real)
    current_Ne = effective_population_size(scenario, t)
    rate = scenario.base_rate

    # Detection occurs when population crosses threshold
    if current_Ne >= scenario.detection_threshold
        # Find approximately when threshold was crossed (crude approximation)
        detection_time = scenario.inflection_time +
                         log((scenario.carrying_capacity - scenario.detection_threshold) /
                             (scenario.detection_threshold - scenario.initial_population)) /
                         scenario.growth_rate

        # Account for detection delay
        investigation_start = detection_time + scenario.detection_delay
        investigation_end = investigation_start + scenario.investigation_duration

        # Reactive sampling during investigation period
        if investigation_start <= t <= investigation_end
            rate *= scenario.reactive_multiplier
        elseif t > investigation_end
            # Continued elevated sampling (but reduced) after investigation
            rate *= 3.0  # moderate ongoing surveillance
        end
    end

    return rate
end
struct SeasonalScenario <: CoalescenceScenario
    sin_divisor::Float64          # divisor in sin(t/divisor) - controls oscillation frequency
    amplitude::Float64            # amplitude multiplier for sin function
    baseline_exp::Float64         # baseline exponent (shifts entire curve up/down)
    sampling_divisor::Float64     # divisor for sampling rate s(t) = n(t)/divisor
end

# Constructor with defaults matching your screenshot
SeasonalScenario(;
    sin_divisor=10.0,
    amplitude=2.0,
    baseline_exp=4.0,
    sampling_divisor=100.0
) = SeasonalScenario(sin_divisor, amplitude, baseline_exp, sampling_divisor)

function effective_population_size(scenario::SeasonalScenario, t::Real)
    # n(t) = exp(sin(t/sin_divisor) * amplitude + baseline_exp)
    return exp(cos(t / scenario.sin_divisor) * scenario.amplitude + scenario.baseline_exp)
end

function sampling_rate(scenario::SeasonalScenario, t::Real)
    # s(t) = n(t)/sampling_divisor
    Ne = effective_population_size(scenario, t)
    return Ne / scenario.sampling_divisor
end
struct LogisticScenario <: CoalescenceScenario
    carrying_capacity::Float64    # maximum population size (K in logistic equation)
    growth_steepness::Float64     # steepness of the logistic curve
    inflection_point::Float64     # time point where growth rate is maximum
    sampling_rate::Float64        # constant sampling rate
 end
 
 # Constructor with defaults matching your screenshot
 LogisticScenario(;
    carrying_capacity = 10000.0,
    growth_steepness = 1.0,       # controls steepness (1 in exp(t-10))
    inflection_point = 10.0,      # inflection point at t=10
    sampling_rate = 20.0          # constant sampling rate
 ) = LogisticScenario(carrying_capacity, growth_steepness, inflection_point, sampling_rate)
 
 function effective_population_size(scenario::LogisticScenario, t::Real)
    # n(t) = carrying_capacity / (1 + exp(growth_steepness * (t - inflection_point)))
    return scenario.carrying_capacity / (1 + exp(scenario.growth_steepness * (t - scenario.inflection_point)))
 end
 
 function sampling_rate(scenario::LogisticScenario, t::Real)
    # Constant sampling rate (from sim_tree(100, n, 20.0))
    return scenario.sampling_rate
 end

 # Save any coalescence scenario to JSON
function save_scenario(scenario::CoalescenceScenario, filename::String)
    # Create a dictionary with type information
    scenario_dict = Dict(
        "type" => string(typeof(scenario)),
        "data" => scenario
    )
    
    # Write to file
    open(filename, "w") do file
        JSON3.pretty(file, scenario_dict)
    end
    
    println("Scenario saved to $filename")
end

# Load a coalescence scenario from JSON
function load_scenario(filename::String)::CoalescenceScenario
    # Read and parse JSON
    json_data = JSON3.read(read(filename, String))
    
    # Extract type and data
    scenario_type = json_data.type
    scenario_data = json_data.data
    
    # Reconstruct the appropriate scenario type
    if scenario_type == "HIVScenario"
        return HIVScenario(
            scenario_data.baseline_Ne,
            scenario_data.treatment_epochs,
            scenario_data.bottleneck_severity,
            scenario_data.resistance_recovery,
            scenario_data.recovery_rate,
            scenario_data.base_rate,
            scenario_data.treatment_failure_boost,
            scenario_data.failure_detection_delay,
            scenario_data.sampling_window
        )
    elseif scenario_type == "InfluenzaScenario"
        return InfluenzaScenario(
            scenario_data.baseline_Ne,
            scenario_data.antigenic_epochs,
            scenario_data.sweep_duration,
            scenario_data.sweep_bottleneck,
            scenario_data.seasonal_amplitude,
            scenario_data.peak_time,
            scenario_data.period,
            scenario_data.base_rate,
            scenario_data.geographic_bias,
            scenario_data.nh_peak_time,
            scenario_data.sh_peak_time,
            scenario_data.surveillance_ramp_up,
            scenario_data.enhancement_factor
        )
    elseif scenario_type == "COVIDScenario"
        return COVIDScenario(
            scenario_data.baseline_Ne,
            scenario_data.variant_emergences,
            scenario_data.growth_advantages,
            scenario_data.displacement_rate,
            scenario_data.global_control_measures,
            scenario_data.base_rate,
            scenario_data.sequencing_ramp_up,
            scenario_data.variant_concern_periods,
            scenario_data.geographic_inequality,
            scenario_data.concern_duration,
            scenario_data.max_sampling_rate
        )
    elseif scenario_type == "LogisticGrowthScenario"
        return LogisticGrowthScenario(
            scenario_data.carrying_capacity,
            scenario_data.growth_rate,
            scenario_data.initial_population,
            scenario_data.inflection_time,
            scenario_data.base_rate,
            scenario_data.detection_threshold,
            scenario_data.detection_delay,
            scenario_data.reactive_multiplier,
            scenario_data.investigation_duration
        )
    elseif scenario_type == "SeasonalScenario"
        return SeasonalScenario(
            scenario_data.sin_divisor,
            scenario_data.amplitude,
            scenario_data.baseline_exp,
            scenario_data.sampling_divisor
        )
    elseif scenario_type == "LogisticScenario"
        return LogisticScenario(
            scenario_data.carrying_capacity,
            scenario_data.growth_steepness,
            scenario_data.inflection_point,
            scenario_data.sampling_rate
        )
    else
        error("Unknown scenario type: $scenario_type")
    end
end