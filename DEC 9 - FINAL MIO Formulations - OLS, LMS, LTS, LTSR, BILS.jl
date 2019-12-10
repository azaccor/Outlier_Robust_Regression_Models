
using CSV, JuMP, MathProgBase, Gurobi, Random, LinearAlgebra, StatsBase, Distributions, Plots, BenchmarkTools, Suppressor

# Formulate optimization problem for Ordinary Least Squares Regression

function OrdinaryLeastSquares(x,y)
#     OLS_model = Model(solver = GurobiSolver(LogFile = append_name,OutputFlag = 0))
    OLS_model = Model(solver = GurobiSolver(OutputFlag = 0))
    
    # Define variables
    n = size(x,1) # observations
    p = size(x,2) # features
    
    # Define decision variables
    @variable(OLS_model, β[1:p+1])

    # Define constraints - none

    # Define objective function
    @objective(OLS_model, Min, sum((y[i] - β[1] - β[2:p+1]' * x[i,:])^2 for i = 1:n))
#     return(OLS_model)
    
    # Run the optimization
    solve(OLS_model)
    betas = getvalue(β)
    objective = getobjectivevalue(OLS_model)
    return(betas)
end

# Formulate optimization problem for Least Quantile Squares Regression
# *********** q: 0, 1, 2, 3, 4

function LeastQuantileSquares(x,y,quantile)
    LQS_model = Model(solver = GurobiSolver(OutputFlag = 0,TimeLimit = 180))
#     LQS_model = Model(solver = GurobiSolver(OutputFlag = 0))
    
    # Define variables
    M = 10000
    n = size(x,1) # observations
    p = size(x,2) # features
    q = ceil(n*(quantile/4))
    
    # Define decision variables
    @variable(LQS_model, γ)
    @variable(LQS_model, z[1:n], Bin)
    @variable(LQS_model, μ[1:n])
    @variable(LQS_model, β[1:p+1])
    @variable(LQS_model, r[1:n])
    @variable(LQS_model, a[1:n])            
            
    # Define constraints
    @constraint(LQS_model, [i=1:n], r[i] == y[i] - β[1] - β[2:p+1]'*x[i,:])
    @constraint(LQS_model, [i=1:n], γ >= a[i] - μ[i])
    @constraint(LQS_model, [i=1:n], a[i] >= r[i])
    @constraint(LQS_model, [i=1:n], a[i] >= -r[i])
    @constraint(LQS_model, [i=1:n], μ[i] <= M*(1 - z[i]))
    @constraint(LQS_model, sum(z[i] for i=1:n) == q)
    @constraint(LQS_model, [i=1:n], μ[i] >= 0)
    
     # Define objective function
    @objective(LQS_model, Min, γ)
#     return(LQS_model)

    # Run the optimization
    solve(LQS_model)
    betas = getvalue(β)
    qth_residual = getvalue(γ)
    return(betas,qth_residual)
end

# Formulate optimization problem for Least Trimmed Squares Regression
# Note: k is a user specified parameter, k <= (n - p - 1)/2

function LeastTrimmedSquares(k,x,y)
    LTS_model = Model(solver = GurobiSolver(OutputFlag = 0,TimeLimit = 500))
    
    # Define variables
    M = 10000
    n = size(x,1) # observations
    p = size(x,2) # features
    
    # Define decision variables
    @variable(LTS_model, z[1:n], Bin)
    @variable(LTS_model, β[1:p+1])
    @variable(LTS_model, r[1:n])
            
    # Define constraints
    @constraint(LTS_model, [i=1:n], β[1] + β[2:p+1]'*x[i,:] - y[i] <= r[i] + M*z[i])
    @constraint(LTS_model, [i=1:n], -β[1] - β[2:p+1]'*x[i,:] + y[i] <= r[i] + M*z[i])
    @constraint(LTS_model, sum(z[i] for i=1:n) <= k)
    @constraint(LTS_model, [i=1:n], r[i] >= 0)
        
    # Define objective function as QMIP ******** (minimizing over β, z)
    @objective(LTS_model, Min, sum(r[i]^2 for i=1:n))
#     return(LTS_model)

    # Run the optimization
    solve(LTS_model)
    betas = getvalue(β)
    outlier_indicators = getvalue(z)
    return(betas, outlier_indicators)
end

# Formulate optimization problem for Least Trimmed Squares Regression
# Note: k is a user specified parameter, k <= (n - p - 1)/2

function LeastTrimmedSumResiduals(k,x,y)
    LTSR_model = Model(solver = GurobiSolver(OutputFlag = 0,TimeLimit = 500))
    
    # Define variables
    M = 10000
    n = size(x,1) # observations
    p = size(x,2) # features
    
    # Define decision variables
    @variable(LTSR_model, z[1:n], Bin)
    @variable(LTSR_model, β[1:p+1])
    @variable(LTSR_model, r[1:n])
            
    # Define constraints
    @constraint(LTSR_model, [i=1:n], β[1] + β[2:p+1]'*x[i,:] - y[i] <= r[i] + M*z[i])
    @constraint(LTSR_model, [i=1:n], -β[1] - β[2:p+1]'*x[i,:] + y[i] <= r[i] + M*z[i])
    @constraint(LTSR_model, sum(z[i] for i=1:n) <= k)
    @constraint(LTSR_model, [i=1:n], r[i] >= 0)
        
    # Define objective function as QMIP ******** (minimizing over β, z)
    @objective(LTSR_model, Min, sum(r[i] for i=1:n))
#     return(LTS_model)

    # Run the optimization
    solve(LTSR_model)
    betas = getvalue(β)
    outlier_indicators = getvalue(z)
    return(betas, outlier_indicators)
end

# Formulate optimization problem for Least Trimmed Squares Regression
# Note: k is a user specified parameter, k <= (n - p - 1)/2

function BoundedInfluence(α,x,y)
    BI_model = Model(solver = GurobiSolver(OutputFlag = 0,TimeLimit = 500))
    
    # Define variables
    M = 10000
    n = size(x,1) # observations
    p = size(x,2) # features
    bkdn = ceil((n-p-1)/2)
    
    # Define parameters
    x_int = hcat(ones(n),x)
    β_ols = (x_int'*x_int)^-1*x_int'*y
    e_residuals = y-x_int*β_ols
    σ_squared = e_residuals'*e_residuals/(n-p+1)
    H = x*(x'*x)^-1*x'
    h = diag(H)
    δ = ((e_residuals).^2/(σ_squared*(1 .- h)))*(1/(p+1))*(h./(1 .- h))
    σ_h = (e_residuals./(1 .- h).^0.5)
    σ_h_squared = (e_residuals./(1 .- h).^0.5).^2
#     println(δ)
    
    # Define decision variables
    @variable(BI_model, z[1:n], Bin)
    @variable(BI_model, β[1:p+1])
    @variable(BI_model, r[1:n])
            
    # Define constraints
    @constraint(BI_model, [i=1:n], β[1] + β[2:p+1]'*x[i,:] - y[i] <= r[i] + M*z[i])
    @constraint(BI_model, [i=1:n], -β[1] - β[2:p+1]'*x[i,:] + y[i] <= r[i] + M*z[i])
    @constraint(BI_model, sum(z[i] for i=1:n) <= bkdn)
    @constraint(BI_model, [i=1:n], r[i] >= 0)
        
    # Define objective function        
    @objective(BI_model, Min, α*sum(r[i]^2 + z[i]*δ[i]^2 for i=1:n) + (1-α)*sum(z[i] for i=1:n))
    
    # Run the optimization
    solve(BI_model)
    betas = getvalue(β)
    outlier_indicators = getvalue(z)
    return(h, betas, outlier_indicators)
end

star_data =  [  1  4.37  5.23   
                2  4.56  5.74   
                3  4.26  4.93
                4  4.56  5.74   
                5  4.30  5.19   
                6  4.46  5.46
                7  3.84  4.65   
                8  4.57  5.27   
                9  4.26  5.57
                10  4.37  5.12  
                11  3.49  5.73  
                12  4.43  5.45
                13  4.48  5.42  
                14  4.01  4.05  
                15  4.29  4.26
                16  4.42  4.58  
                17  4.23  3.94  
                18  4.42  4.18
                19  4.23  4.18  
                20  3.49  5.89  
                21  4.29  4.38
                22  4.29  4.22  
                23  4.42  4.42  
                24  4.49  4.85
                25  4.38  5.02
                26  4.42  4.66 
                27  4.29  4.66
                28  4.38  4.90  
                29  4.22  4.39  
                30  3.48  6.05
                31  4.38  4.42  
                32  4.56  5.10
                33  4.45  5.22
                34  3.49  6.29  
                35  4.23  4.34  
                36  4.62  5.62
                37  4.53  5.10  
                38  4.45  5.22  
                39  4.53  5.18
                40  4.43  5.57  
                41  4.38  4.62  
                42  4.45  5.06
                43  4.50  5.34  
                44  4.45  5.34  
                45  4.55  5.54
                46  4.45  4.98  
                47  4.42  4.50 ]; 

x_star_data = star_data[:,2]
y_star_data = star_data[:,3];

star_betas_OLS = OrdinaryLeastSquares(x_star_data,y_star_data)

star_betas_LQS, star_qth_resid_LQS = LeastQuantileSquares(x_star_data,y_star_data,2)

star_betas_LTS, star_outlier_indicators_LTS = LeastTrimmedSquares(4,x_star_data,y_star_data)

findall(x->x==1, star_outlier_indicators_LTS)

star_betas_LTSR, star_outlier_indicators_LTSR = LeastTrimmedSumResiduals(4,x_star_data,y_star_data)

findall(x->x==1, star_outlier_indicators_LTSR)

h_star, star_betas_BI, star_outlier_indicators_BI = BoundedInfluence(0.4,x_star_data,y_star_data)

findall(x->x==1, star_outlier_indicators_BI)

# Plot all the fits
# gr()
star_data_plot = scatter(x_star_data,y_star_data,series_annotations = text.(1:47, :bottom),
#     title="Fitted Regressions on Hertzsprung-Russell Star Data",
    label="",
    legend=:outerbottom,
    markersize = 2,
    markercolor = :black,
    dpi=300,
    fontfamily="Times",
    size = (600, 500))

f_ols(x_ols) = star_betas_OLS[1] + x_ols.*star_betas_OLS[2]
plot!(f_ols, 3.5, 5,label="Ordinary Least Squares")

f_lqs(x_lqs) = star_betas_LQS[1] + x_lqs.*star_betas_LQS[2]
plot!(f_lqs, 3.5, 5,label="Least Median Squares")

f_lts(x_lts) = star_betas_LTS[1] + x_lts.*star_betas_LTS[2]
plot!(f_lts, 3.5, 5,label="Least Trimmed Squares")

f_ltsr(x_ltsr) = star_betas_LTSR[1] + x_ltsr.*star_betas_LTSR[2]
plot!(f_ltsr, 3.5, 5,label="Least Trimmed Summed Residuals")

f_bi(x_bi) = star_betas_BI[1] + x_bi.*star_betas_BI[2]
plot!(f_bi, 3.5, 5,label="Bounded Influence Least Squares")

savefig("star_data_plot2")

star_data_plot2

stack_loss = [ 1  80  27  89  42
               1  80  27  88  37
               1  75  25  90  37
               1  62  24  87  28
               1  62  22  87  18
               1  62  23  87  18
               1  62  24  93  19
               1  62  24  93  20
               1  58  23  87  15
               1  58  18  80  14
               1  58  18  89  14
               1  58  17  88  13
               1  58  18  82  11
               1  58  19  93  12
               1  50  18  89   8
               1  50  18  86   7
               1  50  19  72   8
               1  50  19  79   8
               1  50  20  80   9
               1  56  20  82  15
               1  70  20  91  15 ];

x_stack_loss = stack_loss[:,2:4];

y_stack_loss = stack_loss[:,5];

stack_loss_betas_OLS = OrdinaryLeastSquares(x_stack_loss,y_stack_loss)

stack_loss_betas_LQS, stack_loss_qth_resid_LQS = LeastQuantileSquares(x_stack_loss,y_stack_loss,2)

stack_loss_betas_LTS, stack_loss_outlier_indicators_LTS = LeastTrimmedSquares(4,x_stack_loss,y_stack_loss)

index = findall(x->x==1, stack_loss_outlier_indicators_LTS)

stack_loss_betas_LTSR, stack_loss_outlier_indicators_LTSR = LeastTrimmedSumResiduals(4,x_stack_loss,y_stack_loss)

findall(x->x==1, stack_loss_outlier_indicators_LTSR)

h_stack_loss, stack_loss_betas_BI, stack_loss_outlier_indicators_BI = BoundedInfluence(0.1,x_stack_loss,y_stack_loss)

findall(x->x==1, stack_loss_outlier_indicators_BI)

h_stack_loss

function standardize(df)
    n, p = size(df)
    for j in 1:p
        mean1 = mean(df[:,j])
        std1 = std(df[:,j])
        for i in 1:n
            df[i,j] = (df[i,j] - mean1)/std1
        end
    end
    return df
end

df1 = CSV.read("sds0.csv", header=false);

y1 = df1[:,1]
X1 = convert(Matrix, df1[:,2:end]);

size(X1)

t_OLS_df1 = @suppress @benchmark df1_betas_OLS = OrdinaryLeastSquares(X1,y1)

time(median(t_OLS_df1))/1000000000

t_LQS_df1 = @suppress @benchmark LeastQuantileSquares(X1,y1,2) 

time(median(t_LQS_df1))/1000000000

t_LTS_df1 = @suppress @benchmark df1_LTS_betas, df1_LTS_outlier_indicators = LeastTrimmedSquares(0,X1,y1)

time(median(t_LTS_df1))/1000000000

t_LTSR_df1 = @suppress @benchmark df1_LTSR_betas, df1_LTSR_outlier_indicators = LeastTrimmedSumResiduals(0,X1,y1)

time(median(t_LTSR_df1))/1000000000

df1_betas_OLS = OrdinaryLeastSquares(X1,y1)

df1_LQS_betas, df1_LQS_qth_resid = LeastQuantileSquares(X1,y1,2)

df1_LTS_betas, df1_LTS_outlier_indicators = LeastTrimmedSquares(0,X1,y1)

df1_LTSR_betas, df1_LTSR_outlier_indicators = LeastTrimmedSumResiduals(0,X1,y1)

df1_BI_betas, df1_BI_outlier_indicators = BoundedInfluence(0.00001,X1,y1)

df2 = CSV.read("sds1.csv", header=false);

y2 = df2[:,1]
X2 = convert(Matrix, df2[:,2:end]);

size(X2)

sum(X2)

mean(X2)

t_OLS_df2 = @suppress @benchmark df2_betas_OLS = OrdinaryLeastSquares(X2,y2)

time(median(t_OLS_df2))/1000000000

t_LQS_df2 = @suppress @benchmark df2_LQS_betas, df2_LQS_qth_resid = LeastQuantileSquares(X2,y2,2)

time(median(t_LQS_df2))/1000000000

t_LTS_df2 = @suppress @benchmark df2_LTS_betas, df2_LTS_outlier_indicators = LeastTrimmedSquares(5,X2,y2)

time(median(t_LTS_df2))/1000000000

t_LTSR_df2 = @suppress @benchmark df2_LTSR_betas, df2_LTSR_outlier_indicators = LeastTrimmedSumResiduals(5,X2,y2)

time(median(t_LTSR_df2))/1000000000

t_BI_df2 = @suppress @benchmark df2_BI_betas, df2_BI_outlier_indicators = BoundedInfluence(0.4,X2,y2)

time(median(t_BI_df2))/1000000000

df2_betas_OLS = OrdinaryLeastSquares(X2,y2)

df2_LQS_betas, df2_LQS_qth_resid = LeastQuantileSquares(X2,y2,2)

df2_LTS_betas, df2_LTS_outlier_indicators = LeastTrimmedSquares(5,X2,y2)

findall(x->x==1, df2_LTS_outlier_indicators)

df2_LTSR_betas, df2_LTSR_outlier_indicators = LeastTrimmedSumResiduals(5,X2,y2)

findall(x->x==1, df2_LTSR_outlier_indicators)

df2_BI_betas, df2_BI_outlier_indicators = BoundedInfluence(0.4,X2,y2)

findall(x->x==1, df2_BI_outlier_indicators)

df3 = CSV.read("sds2.csv", header=false);

y3 = df3[:,1]
X3 = convert(Matrix, df3[:,2:end]);

t_OLS_df3 = @suppress @benchmark df3_betas_OLS = OrdinaryLeastSquares(X3,y3)

time(median(t_OLS_df3))/1000000000

t_LQS_df3 = @suppress @benchmark df3_LQS_betas, df3_LQS_qth_resid = LeastQuantileSquares(X3,y3,2)

time(median(t_LQS_df3))/1000000000

t_LTS_df3 = @suppress @benchmark df3_LTS_betas, df3_LTS_outlier_indicators = LeastTrimmedSquares(9,X3,y3)

time(median(t_LTS_df3))/1000000000

t_LTSR_df3 = @suppress @benchmark df3_LTSR_betas, df3_LTSR_outlier_indicators = LeastTrimmedSumResiduals(9,X3,y3)

time(median(t_LTSR_df3))/1000000000

t_BI_df3 = @suppress @benchmark df3_BI_betas, df3_BI_outlier_indicators = BoundedInfluence(0.82,X3,y3)

time(median(t_BI_df3))/1000000000

df3_betas_OLS = OrdinaryLeastSquares(X3,y3)

df3_LQS_betas, df3_LQS_qth_resid = LeastQuantileSquares(X3,y3,2)

df3_LTS_betas, df3_LTS_outlier_indicators = LeastTrimmedSquares(9,X3,y3)

findall(x->x==1, df3_LTS_outlier_indicators)

df3_LTSR_betas, df3_LTSR_outlier_indicators = LeastTrimmedSumResiduals(9,X3,y3)

findall(x->x==1, df3_LTSR_outlier_indicators)

df3_BI_betas, df3_BI_outlier_indicators = BoundedInfluence(0.82,X3,y3)

findall(x->x==1, df3_BI_outlier_indicators)

df4 = CSV.read("sds3.csv", header=false);

y4 = df4[:,1]
X4 = convert(Matrix, df4[:,2:end]);

size(X4)

t_OLS_df4 = @suppress @benchmark df4_betas_OLS = OrdinaryLeastSquares(X4,y4)

time(median(t_OLS_df4))/1000000000

t_LQS_df4 = @suppress @benchmark df4_LQS_betas, df4_LQS_qth_resid = LeastQuantileSquares(X4,y4,2)

time(median(t_LQS_df4))/1000000000

t_LTS_df4 = @suppress @benchmark df4_LTS_betas, df4_LTS_outlier_indicators = LeastTrimmedSquares(24,X4,y4)

time(median(t_LTS_df4))/1000000000

t_LTSR_df4 = @suppress @benchmark df4_LTSR_betas, df4_LTSR_outlier_indicators = LeastTrimmedSumResiduals(24,X4,y4)

time(median(t_LTSR_df4))/1000000000

t_BI_df4 = @suppress @benchmark df4_BI_betas, df4_BI_outlier_indicators = BoundedInfluence(0.85,X4,y4)

time(median(t_BI_df4))/1000000000

df4_betas_OLS = OrdinaryLeastSquares(X4,y4)

df4_LQS_betas, df4_LQS_qth_resid = LeastQuantileSquares(X4,y4,2)

df4_LTS_betas, df4_LTS_outlier_indicators = LeastTrimmedSquares(24,X4,y4)

findall(x->x==1, df4_LTS_outlier_indicators)

df4_LTSR_betas, df4_LTSR_outlier_indicators = LeastTrimmedSumResiduals(24,X4,y4)

findall(x->x==1, df4_LTSR_outlier_indicators)

df4_BI_betas, df4_BI_outlier_indicators = BoundedInfluence(0.89,X4,y4)

findall(x->x==1, df4_BI_outlier_indicators)

# df5 = CSV.read("./SyntheticData_&_Scripts/df5.csv", header=false)
# y5 = df5[:,1]
# X5 = convert(Matrix, df5[:,2:end]);

# size(X5)

# t_OLS_df5 = @suppress @benchmark df5_betas_OLS = OrdinaryLeastSquares(X5,y5)

# time(median(t_OLS_df5))/1000000000

# t_LQS_df5 = @suppress @benchmark df5_LQS_betas, df5_LQS_qth_resid = LeastQuantileSquares(X5,y5,2)

# time(median(t_LQS_df2))/1000000000

# t_LTS_df5 = @suppress @benchmark df5_LTS_betas, df5_LTS_outlier_indicators = LeastTrimmedSquares(4,X5,y5)

# time(median(t_LTS_df2))/1000000000

# t_ALTS_df5 = @suppress @benchmark df5_LTSR_betas, df5_LTSR_outlier_indicators = LeastTrimmedSumResiduals(4,X5,y5)

# time(median(t_LTSR_df2))/1000000000

# t_BI_df5 = @suppress @benchmark df5_BI_betas, df5_BI_outlier_indicators = BoundedInfluence(0.1,X5,y5)

# time(median(t_BI_df2))/1000000000

# df5_betas_OLS = OrdinaryLeastSquares(X5,y5)

# df5_LQS_betas, df5_LQS_qth_resid = LeastQuantileSquares(X5,y5,2)

# df5_LTS_betas, df5_LTS_outlier_indicators = LeastTrimmedSquares(5,X5,y5)

# findall(x->x==1, df5_LTS_outlier_indicators)

# df5_LTSR_betas, df5_LTSR_outlier_indicators = LeastTrimmedSumResiduals(5,X5,y5)

# findall(x->x==1, df5_LTSR_outlier_indicators)

# df5_BI_betas, df5_BI_outlier_indicators = BoundedInfluence(0.1,X5,y5)

# findall(x->x==1, df5_BI_outlier_indicators)

function standardize(df)
    n, p = size(df)
    for j in 1:p
        mean1 = mean(df[:,j])
        std1 = std(df[:,j])
        for i in 1:n
            df[i,j] = (df[i,j] - mean1)/std1
        end
    end
    return df
end

aq = CSV.read("Results/qsar_aquatic_toxicity.csv", header=false)
#Convert to matrices
Random.seed!(95)
aq = convert(Matrix, aq)
aq = aq[shuffle(1:end), :]

size(aq)

aq = standardize(aq);

# Split into 50%, 25%, and 25%
train_x = aq[1:40, 4:8]
train_y = aq[1:40, 9]

valid_x = aq[41:60, 4:8]
valid_y = aq[41:60, 9]

test_x = aq[61:80, 4:8]
test_y = aq[61:80, 9];

betas_OLS = OrdinaryLeastSquares(train_x,train_y)

betas_LQS, LQS_qth_resid = LeastQuantileSquares(train_x,train_y,2)

betas_LTS, LTS_outlier_indicators = LeastTrimmedSquares(4,train_x,train_y)

betas_LTSR, LTSR_outlier_indicators = LeastTrimmedSumResiduals(4,train_x,train_y)

betas_BI, BI_outlier_indicators = BoundedInfluence(0.4,train_x,train_y)

mean((valid_y .- betas_OLS[1] .- valid_x*betas_OLS[2:end]).^2)
mean((valid_y .- betas_LQS[1] .- valid_x*betas_LQS[2:end]).^2)
mean((valid_y .- betas_LTS[1] .- valid_x*betas_LTS[2:end]).^2)
mean((valid_y .- betas_LTSR[1] .- valid_x*betas_LTSR[2:end]).^2)
mean((valid_y .- betas_BI[1] .- valid_x*betas_BI[2:end]).^2)

full_x = vcat(train_x, valid_x)
full_y = vcat(train_y, valid_y);

time_OLS = @suppress @benchmark OrdinaryLeastSquares(full_x,full_y)
time(median(time_OLS))/1000000000

time_LQS = @suppress @benchmark betas_LQS, LQS_qth_resid = LeastQuantileSquares(full_x,full_y,2)
time(median(time_LQS))/1000000000

time_LTS = @suppress @benchmark  LeastTrimmedSquares(4,full_x,full_y)
time(median(time_LTS))/1000000000

time_LTSR = @suppress @benchmark LeastTrimmedSumResiduals(4,full_x,full_y)
time(median(time_LTSR))/1000000000

time_BI = @suppress @benchmark BoundedInfluence(0.4,full_x,full_y)
time(median(time_BI))/1000000000
