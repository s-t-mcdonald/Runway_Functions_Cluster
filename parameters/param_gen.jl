
using CSV, Base.Threads, DataFrames, Random

AIRPORTS = [
    "katl",
    "kclt", 
    "kden",
    "kdfw",
    "kjfk",
    "kmem",
    "kmia",
    "kord",
    "kphx",
    "ksea"
]
LOOKAHEAD = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]


# Modeling Parameters
DATA            = ["PD", "SM", "PD_SM"]
CONFIG_SUPPORT  = [0, 1]

EPOCHS          = [300]
NUMBER_TRIALS   = [100]
PATIENCE        = [10]

param_df    = DataFrame(PARAM = Int64[], AIRPORTS = String[], LOOKAHEAD = Int64[], DATA = String[], CONFIG_SUPPORT = Int64[],
                        EPOCHS = Int64[], NUMBER_TRIALS = Int64[], PATIENCE = Int64[])

PARAMS      = vcat(collect(Iterators.product(AIRPORTS, LOOKAHEAD, DATA, CONFIG_SUPPORT, EPOCHS, NUMBER_TRIALS, PATIENCE))...)

k = 1
for p_num ∈ 1:length(PARAMS)
    global k
    
    param_array = [p for p in PARAMS[p_num]]
    pushfirst!(param_array, k)
    push!(param_df, param_array)

    k = k + 1
end

CSV.write("parameters/parameter_array.csv", param_df)