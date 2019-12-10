
using JuMP, IJulia, Gurobi, DataFrames, CSV, Random, LinearAlgebra, Distributions

function generate_synthetic_data(β0, β1; SNR = 30)
    Random.seed!(99)
    X1 = rand(40,4) 
    Y1 = β0 .+ X1 * β1 
    noise = randn(40)
    k = sqrt(var(Y1) / (SNR*var(noise)))
    Y_with_noise = Y1 + k.*noise
    df = hcat(Y_with_noise,X1)
    return df
end

function create_data(n, p, contam_prct, sig_noise_ratio)
    Random.seed!(99)
    ## Create indices to contaminate X and y
    contam = randperm(n*(p+1))[1:Int(round(n*contam_prct))]
#     X = []
#     y = []
    
    ## Create X vars where the mean and sd of each successive column increases (no real reason)
    ## If the count of the nested for loop iterator == anything in contam indices, add from U(80,200)
    count = 0
    X = randn(n,p)
#     ## Reshape to create matrix
#     X = convert(Matrix, reshape(X, n, p))
    
    ## Create true 𝛽 values
    if p == 4
        𝛽 = [0.4, -0.5, 0.8, 0.5]
    elseif p == 5 && contam_prct == .12
        𝛽 = [0.3, -0.6, 1, 0.7, -0.5]
    elseif p == 5 && contam_prct == .24
        𝛽 = [0.3, -0.6, 1.3, 0.7, -0.7] 
    elseif p == 6
        𝛽 = [0.3, -0.4, 0.6, 0.4, -0.6, 0.6]
    end
    
    
    y_pre_noise = X*𝛽 
#     @show y
    noise = randn(n)
    e = sqrt(var(y_pre_noise)/(sig_noise_ratio*var(noise)))
    @show e
    y = y_pre_noise + e.*noise
#     @show y
    
    for i in 1:n
        count += 1
        if count in contam
            y[i] += rand(Uniform(6, 10), 1)[1]
        else
            y[i] = y[i]
        end
    end
    
    ## Perturb some of the X values now that Y has been created uncontaminated
    X = vec(reshape(X, n*p, 1))
    count2 = 0
    for i in 1:(n*p)
        count2 += 1
        if count2 in contam
            X[i] += rand(Uniform(4,6), 1)[1]
        end 
    end
    X = convert(Matrix, reshape(X, n, p))
    
    ## Store outlier location in row, col form:
    contam_ind = sort(contam)
    cols = []
    rows = []
    for i in 1:length(contam_ind)
        append!(cols, ceil(Int, contam_ind[i]/n))
        if contam_ind[i] % n == 0
            append!(rows, n)
        else
            append!(rows, contam_ind[i] % n)
        end
    end
    outliers = hcat(cols, rows)
    
    return X, y, 𝛽, contam, outliers
end

function get_outliers(contam, n, p)
    ## Be able to return the indices of outliers
    contam_ind = sort(contam)
    cols = []
    rows = []
    for i in 1:length(contam_ind)
        append!(cols, ceil(Int, contam_ind[i]/n))
        if contam_ind[i] % n == 0
            append!(rows, n)
        else
            append!(rows, contam_ind[i] % n)
        end
    end

    for i in 1:length(contam_ind)
        if cols[i] <= p
            println("Outlier ", i, " in X", cols[i], ", row ", rows[i])
        else
            println("Outlier ", i, " in Y, row ", rows[i])
        end
    end
end

X1, y1, 𝛽1, contam1, outliers1 = create_data(25, 4, 0, 40);
X2, y2, 𝛽2, contam2, outliers2 = create_data(40, 5, 0.12, 40);
X3, y3, 𝛽3, contam3, outliers3 = create_data(40, 5, 0.24, 40);
X4, y4, 𝛽4, contam4, outliers4 = create_data(80, 6, 0.36, 40);

#get_outliers(contam1, 25, 4)
#get_outliers(contam2, 40, 5)
get_outliers(contam3, 40, 5)
#get_outliers(contam4, 80, 6)

# Test:
mean(y3 - X3*𝛽3)

sd_X1 = map(std, (X1[:,j] for j=1:size(X1,2)))
sd_X2 = map(std, (X2[:,j] for j=1:size(X2,2)))
sd_X3 = map(std, (X3[:,j] for j=1:size(X3,2)))
sd_X4 = map(std, (X4[:,j] for j=1:size(X4,2)))

𝛽1s = 𝛽1.*(sd_X1/std(y1))
𝛽2s = 𝛽2.*(sd_X2/std(y2))
𝛽3s = 𝛽3.*(sd_X3/std(y3))
𝛽4s = 𝛽4.*(sd_X4/std(y4));

## Create dataframes with X and y
df1 = hcat(y1, X1);
df2 = hcat(y2, X2);
df3 = hcat(y3, X3);
df4 = hcat(y4, X4);
#df5 = hcat(y5, X5);

CSV.write("Data/df1.csv", DataFrame(df1); writeheader=false);
CSV.write("Data/df2.csv", DataFrame(df2); writeheader=false);
CSV.write("Data/df3.csv", DataFrame(df3); writeheader=false);
CSV.write("Data/df4.csv", DataFrame(df4); writeheader=false);
#CSV.write("Data/df5.csv", DataFrame(df5); writeheader=false);
