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
    baseline_Ne = 500,
    treatment_epochs = [0.3, 0.7],
    bottleneck_severity = 0.1,
    resistance_recovery = 2.0,
    recovery_rate = 1.5,
    base_rate = 0.02,
    treatment_failure_boost = 15.0,
    failure_detection_delay = 0.1,
    sampling_window = 0.2
) = HIVScenario(baseline_Ne, treatment_epochs, bottleneck_severity, 
                resistance_recovery, recovery_rate, base_rate, 
                treatment_failure_boost, failure_detection_delay, sampling_window)

InfluenzaScenario(;
    baseline_Ne = 2000,
    antigenic_epochs = [0.25, 0.6],
    sweep_duration = 0.15,
    sweep_bottleneck = 0.2,
    seasonal_amplitude = 0.6,
    peak_time = 0.25,
    period = 1.0,
    base_rate = 0.15,
    geographic_bias = 2.0,
    nh_peak_time = 0.25,
    sh_peak_time = 0.75,
    surveillance_ramp_up = [0.1, 0.4],
    enhancement_factor = 3.0
) = InfluenzaScenario(baseline_Ne, antigenic_epochs, sweep_duration, sweep_bottleneck,
                      seasonal_amplitude, peak_time, period, base_rate, geographic_bias,
                      nh_peak_time, sh_peak_time, surveillance_ramp_up, enhancement_factor)

COVIDScenario(;
    baseline_Ne = 5000,
    variant_emergences = [0.15, 0.4, 0.65],
    growth_advantages = [1.5, 2.0, 2.5],
    displacement_rate = 4.0,
    global_control_measures = [0.1, 0.8],
    base_rate = 0.05,
    sequencing_ramp_up = 0.2,
    variant_concern_periods = [0.15, 0.4, 0.65],
    geographic_inequality = 3.0,
    concern_duration = 0.1,
    max_sampling_rate = 2.0
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
    seasonal_factor = 1 + scenario.seasonal_amplitude * cos(2π * (t/scenario.period - scenario.peak_time))
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
    nh_factor = 1 + scenario.geographic_bias * exp(cos(2π * (t/scenario.period - scenario.nh_peak_time)))
    sh_factor = 1 + 0.3 * exp(cos(2π * (t/scenario.period - scenario.sh_peak_time)))
    
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

# Usage examples:
# hiv = HIVScenario()  # uses all defaults
# hiv_custom = HIVScenario(baseline_Ne=1000, treatment_epochs=[0.2, 0.5, 0.8])
# Ne_t = effective_population_size(hiv, 0.5)
# sample_rate_t = sampling_rate(hiv, 0.5)