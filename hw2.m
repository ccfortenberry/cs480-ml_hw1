# Curtis Fortenberry
# CS480-F
# hw1 problem 2
# Shawn Butler

# Read training data from csv
wine_data = csvread ("./WineData/wine_train_set.csv",1,0);

fixed_acidity        = wine_data(:,1);
volatile_acidity     = wine_data(:,2);
citric_acid          = wine_data(:,3);
residual_sugar       = wine_data(:,4);
chlorides            = wine_data(:,5);
free_sulfur_dioxide  = wine_data(:,6);
total_sulfur_dioxide = wine_data(:,7);
density              = wine_data(:,8);
pH                   = wine_data(:,9);
sulphates            = wine_data(:,10);
alcohol              = wine_data(:,11);
quality              = wine_data(:,12);

################################################################################
# Visualize the data
figure(1);
subplot (3,4,1);
plot (quality, fixed_acidity, '.');
xlabel("Quality");
ylabel("Fixed Acidity");
title("F. Acidity vs Quality");

subplot (3,4,2);
plot (quality, volatile_acidity, '.');
xlabel("Quality");
ylabel("Volatile Acidity");
title("V. Acidity vs Quality");

subplot (3,4,3);
plot (quality, citric_acid, '.');
xlabel("Quality");
ylabel("Citric Acid");
title("Citric Acid vs Quality");

subplot (3,4,4);
plot (quality, residual_sugar, '.');
xlabel("Quality");
ylabel("Residual Sugar");
title("Residual Sugar vs Quality");

subplot (3,4,5);
plot (quality, chlorides, '.');
xlabel("Quality");
ylabel("Chlorides");
title("Chlorides vs Quality");

subplot (3,4,6);
plot (quality, free_sulfur_dioxide, '.');
xlabel("Quality");
ylabel("Free Sulfur Dioxied");
title("FSD vs Quality");

subplot (3,4,7);
plot (quality, total_sulfur_dioxide, '.');
xlabel("Quality");
ylabel("Total Sulfur Dioxide");
title("TSD vs Quality");

subplot (3,4,8);
plot (quality, density, '.');
xlabel("Quality");
ylabel("Density");
title("Density vs Quality");

subplot (3,4,9);
plot (quality, pH, '.');
xlabel("Quality");
ylabel("pH");
title("pH vs Quality");

subplot (3,4,10);
plot (quality, sulphates, '.');
xlabel("Quality");
ylabel("Sulphates");
title("Sulphates vs Quality");

subplot (3,4,11);
plot (quality, alcohol, '.');
xlabel("Quality");
ylabel("Alcohol");
title("Alcohol vs Quality");

################################################################################
# Calculate the priors for each class of Y
# Where Ysub(k) is wine quality on a scale of [1,9]
priors = [0;0;0;0;0;0;0;0;0];

# We'll also snag the sum for each attribute while we're iterating through
# the data set, each with a vector that is indexed by wine quality
total_fixed_acidity        = [0;0;0;0;0;0;0;0;0];
total_volatile_acidity     = [0;0;0;0;0;0;0;0;0];
total_citric_acid          = [0;0;0;0;0;0;0;0;0];
total_residual_sugar       = [0;0;0;0;0;0;0;0;0];
total_chlorides            = [0;0;0;0;0;0;0;0;0];
total_free_sulfur_dioxide  = [0;0;0;0;0;0;0;0;0];
total_total_sulfur_dioxide = [0;0;0;0;0;0;0;0;0];
total_density              = [0;0;0;0;0;0;0;0;0];
total_pH                   = [0;0;0;0;0;0;0;0;0];
total_sulphates            = [0;0;0;0;0;0;0;0;0];
total_alcohol              = [0;0;0;0;0;0;0;0;0];

for i = 1:rows(wine_data)
  priors(quality(i)) = priors(quality(i)) + 1;
  
  total_fixed_acidity(quality(i)) = total_fixed_acidity(quality(i)) + fixed_acidity(i);
  total_volatile_acidity(quality(i)) = total_volatile_acidity(quality(i)) + volatile_acidity(i);
  total_citric_acid(quality(i)) = total_citric_acid(quality(i)) + citric_acid(i);
  total_residual_sugar(quality(i)) = total_residual_sugar(quality(i)) + residual_sugar(i);
  total_chlorides(quality(i)) = total_chlorides(quality(i)) + chlorides(i);
  total_free_sulfur_dioxide(quality(i)) = total_free_sulfur_dioxide(quality(i)) + free_sulfur_dioxide(i);
  total_total_sulfur_dioxide(quality(i)) = total_total_sulfur_dioxide(quality(i)) + total_sulfur_dioxide(i);
  total_density(quality(i)) = total_density(quality(i)) + density(i);
  total_pH(quality(i)) = total_pH(quality(i)) + pH(i);
  total_sulphates(quality(i)) = total_sulphates(quality(i)) + sulphates(i);
  total_alcohol(quality(i)) = total_alcohol(quality(i)) + alcohol(i);
endfor

total_quality = priors; # We'll save this container as it is to calculate
                        # the means and variances later
priors = priors ./ length(quality);

disp("Class probabilities of wine quality[1-9]");
for i = 1:rows(priors)
  printf("[%i]: %f\n", i, priors(i));
endfor

disp("");

# We model using a Gaussian distribution
mu_fixed_acidity        = [0;0;0;0;0;0;0;0;0];
mu_volatile_acidity     = [0;0;0;0;0;0;0;0;0];
mu_citric_acid          = [0;0;0;0;0;0;0;0;0];
mu_residual_sugar       = [0;0;0;0;0;0;0;0;0];
mu_chlorides            = [0;0;0;0;0;0;0;0;0];
mu_free_sulfur_dioxide  = [0;0;0;0;0;0;0;0;0];
mu_total_sulfur_dioxide = [0;0;0;0;0;0;0;0;0];
mu_density              = [0;0;0;0;0;0;0;0;0];
mu_pH                   = [0;0;0;0;0;0;0;0;0];
mu_sulphates            = [0;0;0;0;0;0;0;0;0];
mu_alcohol              = [0;0;0;0;0;0;0;0;0];

# Iterate and calculate the means
# It doesn't matter which mu_vector length we use since all = 9
# This one just came up on autocomplete first... could just use i = 1:9
# Either way. Some say potato, i say lumpy brown apple
for i = 1:rows(mu_alcohol)  
  if (total_quality(i) != 0)
    mu_fixed_acidity(i) = total_fixed_acidity(i) / total_quality(i);
    mu_volatile_acidity(i) = total_volatile_acidity(i) / total_quality(i);
    mu_citric_acid(i) = total_citric_acid(i) / total_quality(i);
    mu_residual_sugar(i) = total_residual_sugar(i) / total_quality(i);
    mu_chlorides(i) = total_chlorides(i) / total_quality(i);
    mu_free_sulfur_dioxide(i) = total_free_sulfur_dioxide(i) / total_quality(i);
    mu_total_sulfur_dioxide(i) = total_total_sulfur_dioxide(i) / total_quality(i);
    mu_density(i) = total_density(i) / total_quality(i);
    mu_pH(i) = total_pH(i) / total_quality(i);
    mu_sulphates(i) = total_sulphates(i) / total_quality(i);
    mu_alcohol(i) = total_alcohol(i) / total_quality(i);
  endif
endfor

disp("Gaussian means of attributes");
for i = 1:rows(mu_chlorides)
  printf("Fixed Acidity[%i]:        | %f\n", i, mu_fixed_acidity(i));
  printf("Volatile Acidity[%i]:     | %f\n", i, mu_volatile_acidity(i));
  printf("Citric Acid[%i]:          | %f\n", i, mu_citric_acid(i));
  printf("Residual Sugar[%i]:       | %f\n", i, mu_residual_sugar(i));
  printf("Chlorides[%i]:            | %f\n", i, mu_chlorides(i));
  printf("Free Sulfur Dioxide[%i]:  | %f\n", i, mu_free_sulfur_dioxide(i));
  printf("Total Sulfur Dioxide[%i]: | %f\n", i, mu_total_sulfur_dioxide(i));
  printf("Density[%i]:              | %f\n", i, mu_density(i));
  printf("pH[%i]:                   | %f\n", i, mu_pH(i));
  printf("Sulphates[%i]:            | %f\n", i, mu_sulphates(i));
  printf("Alcohol[%i]:              | %f\n", i, mu_alcohol(i));
endfor
disp("");

# Second verse same as the first, except here we calculate the variances
# sqdiffsum will be (att(i) - mu(i))^2, which is what we use to calculate
# Gaussian variance
sqdiffsum_fixed_acidity        = [0;0;0;0;0;0;0;0;0];
sqdiffsum_volatile_acidity     = [0;0;0;0;0;0;0;0;0];
sqdiffsum_citric_acid          = [0;0;0;0;0;0;0;0;0];
sqdiffsum_residual_sugar       = [0;0;0;0;0;0;0;0;0];
sqdiffsum_chlorides            = [0;0;0;0;0;0;0;0;0];
sqdiffsum_free_sulfur_dioxide  = [0;0;0;0;0;0;0;0;0];
sqdiffsum_total_sulfur_dioxide = [0;0;0;0;0;0;0;0;0];
sqdiffsum_density              = [0;0;0;0;0;0;0;0;0];
sqdiffsum_pH                   = [0;0;0;0;0;0;0;0;0];
sqdiffsum_sulphates            = [0;0;0;0;0;0;0;0;0];
sqdiffsum_alcohol              = [0;0;0;0;0;0;0;0;0];

for i = 1:rows(wine_data)
  sqdiffsum_fixed_acidity(quality(i)) = sqdiffsum_fixed_acidity(quality(i)) + (fixed_acidity(i) - mu_fixed_acidity(quality(i)))^2;
  sqdiffsum_volatile_acidity(quality(i)) = sqdiffsum_volatile_acidity(quality(i)) + (volatile_acidity(i) - mu_volatile_acidity(quality(i)))^2;
  sqdiffsum_citric_acid(quality(i)) = sqdiffsum_citric_acid(quality(i)) + (citric_acid(i) - mu_citric_acid(quality(i)))^2;
  sqdiffsum_residual_sugar(quality(i)) = sqdiffsum_residual_sugar(quality(i)) + (residual_sugar(i) - mu_residual_sugar(quality(i)))^2;
  sqdiffsum_chlorides(quality(i)) = sqdiffsum_chlorides(quality(i)) + (chlorides(i) - mu_chlorides(quality(i)))^2;
  sqdiffsum_free_sulfur_dioxide(quality(i)) = sqdiffsum_free_sulfur_dioxide(quality(i)) + (free_sulfur_dioxide(i) - mu_free_sulfur_dioxide(quality(i)))^2;
  sqdiffsum_total_sulfur_dioxide(quality(i)) = sqdiffsum_total_sulfur_dioxide(quality(i)) + (total_sulfur_dioxide(i) - mu_total_sulfur_dioxide(quality(i)))^2;
  sqdiffsum_density(quality(i)) = sqdiffsum_density(quality(i)) + (density(i) - mu_density(quality(i)))^2;
  sqdiffsum_pH(quality(i)) = sqdiffsum_pH(quality(i)) + (pH(i) - mu_pH(quality(i)))^2;
  sqdiffsum_sulphates(quality(i)) = sqdiffsum_sulphates(quality(i)) + (sulphates(i) - mu_sulphates(quality(i)))^2;
  sqdiffsum_alcohol(quality(i)) = sqdiffsum_alcohol(quality(i)) + (alcohol(i) - mu_alcohol(quality(i)))^2;
endfor

sigma_fixed_acidity        = [0;0;0;0;0;0;0;0;0];
sigma_volatile_acidity     = [0;0;0;0;0;0;0;0;0];
sigma_citric_acid          = [0;0;0;0;0;0;0;0;0];
sigma_residual_sugar       = [0;0;0;0;0;0;0;0;0];
sigma_chlorides            = [0;0;0;0;0;0;0;0;0];
sigma_free_sulfur_dioxide  = [0;0;0;0;0;0;0;0;0];
sigma_total_sulfur_dioxide = [0;0;0;0;0;0;0;0;0];
sigma_density              = [0;0;0;0;0;0;0;0;0];
sigma_pH                   = [0;0;0;0;0;0;0;0;0];
sigma_sulphates            = [0;0;0;0;0;0;0;0;0];
sigma_alcohol              = [0;0;0;0;0;0;0;0;0];

for i = 1:rows(sigma_alcohol)
  if (total_quality(i) != 0)
    sigma_fixed_acidity(i) = sqdiffsum_fixed_acidity(i) / (total_quality(i) - 1);
    sigma_volatile_acidity(i) = sqdiffsum_volatile_acidity(i) / (total_quality(i) - 1);
    sigma_citric_acid(i) = sqdiffsum_citric_acid(i) / (total_quality(i) - 1);
    sigma_residual_sugar(i) = sqdiffsum_residual_sugar(i) / (total_quality(i) - 1);
    sigma_chlorides(i) = sqdiffsum_chlorides(i) / (total_quality(i) - 1);
    sigma_free_sulfur_dioxide(i) = sqdiffsum_free_sulfur_dioxide(i) / (total_quality(i) - 1);
    sigma_total_sulfur_dioxide(i) = sqdiffsum_total_sulfur_dioxide(i) / (total_quality(i) - 1);
    sigma_density(i) = sqdiffsum_density(i) / (total_quality(i) - 1);
    sigma_pH(i) = sqdiffsum_pH(i) / (total_quality(i) - 1);
    sigma_sulphates(i) = sqdiffsum_sulphates(i) / (total_quality(i) - 1);
    sigma_alcohol(i) = sqdiffsum_alcohol(i) / (total_quality(i) - 1);  
  endif
endfor

# Zero check
for i = 1:rows(sigma_fixed_acidity)
  if (sigma_fixed_acidity(i)==0) sigma_fixed_acidity(i) = 1; endif
  if (sigma_volatile_acidity(i)==0) sigma_volatile_acidity(i) = 1; endif
  if (sigma_citric_acid(i)==0) sigma_citric_acid(i) = 1; endif
  if (sigma_residual_sugar(i)==0) sigma_residual_sugar(i) = 1; endif
  if (sigma_chlorides(i)==0) sigma_chlorides(i) = 1; endif
  if (sigma_free_sulfur_dioxide(i)==0) sigma_free_sulfur_dioxide(i) = 1; endif
  if (sigma_total_sulfur_dioxide(i)==0) sigma_total_sulfur_dioxide(i) = 1; endif
  if (sigma_density(i)==0) sigma_density(i) = 1; endif
  if (sigma_pH(i)==0) sigma_pH(i) = 1; endif
  if (sigma_sulphates(i)==0) sigma_sulphates(i) = 1; endif
  if (sigma_alcohol(i)==0) sigma_alcohol(i) = 1; endif
endfor

disp("Gaussian standard deviation of attributes");
for i = 1:rows(sigma_chlorides)
  printf("Fixed Acidity[%i]:        | %f\n", i, sqrt (sigma_fixed_acidity(i)));
  printf("Volatile Acidity[%i]:     | %f\n", i, sqrt (sigma_volatile_acidity(i)));
  printf("Citric Acid[%i]:          | %f\n", i, sqrt (sigma_citric_acid(i)));
  printf("Residual Sugar[%i]:       | %f\n", i, sqrt (sigma_residual_sugar(i)));
  printf("Chlorides[%i]:            | %f\n", i, sqrt (sigma_chlorides(i)));
  printf("Free Sulfur Dioxide[%i]:  | %f\n", i, sqrt (sigma_free_sulfur_dioxide(i)));
  printf("Total Sulfur Dioxide[%i]: | %f\n", i, sqrt (sigma_total_sulfur_dioxide(i)));
  printf("Density[%i]:              | %f\n", i, sqrt (sigma_density(i)));
  printf("pH[%i]:                   | %f\n", i, sqrt (sigma_pH(i)));
  printf("Sulphates[%i]:            | %f\n", i, sqrt (sigma_sulphates(i)));
  printf("Alcohol[%i]:              | %f\n", i, sqrt (sigma_alcohol(i)));
endfor
disp("");

# Evaluating the model with the Training Data
class_probabilities = [0;0;0;0;0;0;0;0;0];
likelihoods         = [0;0;0;0;0;0;0;0;0];
probabilities_fixed_acidity        = [0;0;0;0;0;0;0;0;0];
probabilities_volatile_acidity     = [0;0;0;0;0;0;0;0;0];
probabilities_citric_acid          = [0;0;0;0;0;0;0;0;0];
probabilities_residual_sugar       = [0;0;0;0;0;0;0;0;0];
probabilities_chlorides            = [0;0;0;0;0;0;0;0;0];
probabilities_free_sulfur_dioxide  = [0;0;0;0;0;0;0;0;0];
probabilities_total_sulfur_dioxide = [0;0;0;0;0;0;0;0;0];
probabilities_density              = [0;0;0;0;0;0;0;0;0];
probabilities_pH                   = [0;0;0;0;0;0;0;0;0];
probabilities_sulphates            = [0;0;0;0;0;0;0;0;0];
probabilities_alcohol              = [0;0;0;0;0;0;0;0;0];

guesses  = 0;
correct  = 0;
accuracy = 0;

disp("Evaluating training set:\n______________________");
for i = 1:rows(wine_data)
  for j = 1:rows(priors)
    probabilities_fixed_acidity(j) = (1 / sqrt((2*pi*sigma_fixed_acidity(j)))) * exp (-0.5*((fixed_acidity(i) - mu_fixed_acidity(j))^2 / sigma_fixed_acidity(j)));
    if (probabilities_fixed_acidity(j)==0) probabilities_fixed_acidity(j) = realmin; endif
    
    probabilities_volatile_acidity(j) = (1 / sqrt((2*pi*sigma_volatile_acidity(j)))) * exp (-0.5*((volatile_acidity(i) - mu_volatile_acidity(j))^2 / sigma_volatile_acidity(j)));
    if (probabilities_volatile_acidity(j)==0) probabilities_volatile_acidity(j) = realmin; endif
    
    probabilities_citric_acid(j) = (1 / sqrt((2*pi*sigma_citric_acid(j)))) * exp (-0.5*((citric_acid(i) - mu_citric_acid(j))^2 / sigma_citric_acid(j)));
    if (probabilities_citric_acid(j)==0) probabilities_citric_acid(j) = realmin; endif
    
    probabilities_residual_sugar(j) = (1 / sqrt((2*pi*sigma_residual_sugar(j)))) * exp (-0.5*((residual_sugar(i) - mu_residual_sugar(j))^2 / sigma_residual_sugar(j)));
    if (probabilities_residual_sugar(j)==0) probabilities_residual_sugar(j) = realmin; endif
    
    probabilities_chlorides(j) = (1 / sqrt((2*pi*sigma_chlorides(j)))) * exp (-0.5*((chlorides(i) - mu_chlorides(j))^2 / sigma_chlorides(j)));
    if (probabilities_chlorides(j)==0) probabilities_chlorides(j) = realmin; endif
    
    probabilities_free_sulfur_dioxide(j) = (1 / sqrt((2*pi*sigma_free_sulfur_dioxide(j)))) * exp (-0.5*((free_sulfur_dioxide(i) - mu_free_sulfur_dioxide(j))^2 / sigma_free_sulfur_dioxide(j)));
    if (probabilities_free_sulfur_dioxide(j)==0) probabilities_free_sulfur_dioxide(j) = realmin; endif
    
    probabilities_total_sulfur_dioxide(j) = (1 / sqrt((2*pi*sigma_total_sulfur_dioxide(j)))) * exp (-0.5*((total_sulfur_dioxide(i) - mu_total_sulfur_dioxide(j))^2 / sigma_total_sulfur_dioxide(j)));
    if (probabilities_total_sulfur_dioxide(j)==0) probabilities_total_sulfur_dioxide(j) = realmin; endif
    
    probabilities_density(j) = (1 / sqrt((2*pi*sigma_density(j)))) * exp (-0.5*((density(i) - mu_density(j))^2 / sigma_density(j)));
    if (probabilities_density(j)==0) probabilities_density(j) = realmin; endif
    
    probabilities_pH(j) = (1 / sqrt((2*pi*sigma_pH(j)))) * exp (-0.5*((pH(i) - mu_pH(j))^2 / sigma_pH(j)));
    if (probabilities_pH(j)==0) probabilities_pH(j) = realmin; endif
    
    probabilities_sulphates(j) = (1 / sqrt((2*pi*sigma_sulphates(j)))) * exp (-0.5*((sulphates(i) - mu_sulphates(j))^2 / sigma_sulphates(j)));
    if (probabilities_sulphates(j)==0) probabilities_sulphates(j) = realmin; endif
    
    probabilities_alcohol(j) = (1 / sqrt((2*pi*sigma_alcohol(j)))) * exp (-0.5*((alcohol(i) - mu_alcohol(j))^2 / sigma_alcohol(j)));
    if (probabilities_alcohol(j)==0) probabilities_alcohol(j) = realmin; endif
  endfor
  
  p_x_given_a = [0;0;0;0;0;0;0;0;0];
  for p = 1:rows(p_x_given_a)
    p_x_given_a(p) = probabilities_fixed_acidity(p) * probabilities_volatile_acidity(p) * probabilities_citric_acid(p) * probabilities_residual_sugar(p) * probabilities_chlorides(p) * probabilities_free_sulfur_dioxide(p) * probabilities_total_sulfur_dioxide(p) * probabilities_density(p) * probabilities_pH(p) * probabilities_sulphates(p) * probabilities_alcohol(p);
  endfor
  
  for k = 1:rows(likelihoods)
    likelihoods(k) = priors(k) * p_x_given_a(k);
  endfor
  
  prediction = 0;
  highest_probability = 0;
  
  for n = 1:rows(class_probabilities)
    class_probabilities(n) = likelihoods(n) / sum (likelihoods);
    y_sub_k = class_probabilities(n);
    if (y_sub_k > highest_probability)
      highest_probability = y_sub_k;
      prediction = n;
    endif
  endfor
  
  if (prediction == quality(i))
    correct = correct + 1;
    guesses = guesses + 1;
  else
    guesses = guesses + 1;
  endif
  
  accuracy = correct / guesses * 100;
  
  printf("Prediction: %i, Actual: %i\nAccuracy = %f\n_______________________\n", prediction, quality(i), accuracy);
endfor

printf("Overall Naive Bayes Model accuracy: %f\n", accuracy);
disp("");

################################################################################
# We only achieve approx. 45.02% accuracy
# Let's do some analysis of correlations to see if there are attributes
# that can be removed

corr_fixed_acidity = [corr(fixed_acidity, fixed_acidity),
                      corr(fixed_acidity, volatile_acidity),
                      corr(fixed_acidity, citric_acid),
                      corr(fixed_acidity, residual_sugar),
                      corr(fixed_acidity, chlorides),
                      corr(fixed_acidity, free_sulfur_dioxide),
                      corr(fixed_acidity, total_sulfur_dioxide),
                      corr(fixed_acidity, density),
                      corr(fixed_acidity, pH),
                      corr(fixed_acidity, sulphates),
                      corr(fixed_acidity, alcohol)];

figure(2);
subplot(3,4,1);
bar (corr_fixed_acidity);
xlabel("Attributes");
ylabel("Correlation");
title("Fixed Acidity");

corr_volatile_acidity = [corr(volatile_acidity, fixed_acidity),
                         corr(volatile_acidity, volatile_acidity),
                         corr(volatile_acidity, citric_acid),
                         corr(volatile_acidity, residual_sugar),
                         corr(volatile_acidity, chlorides),
                         corr(volatile_acidity, free_sulfur_dioxide),
                         corr(volatile_acidity, total_sulfur_dioxide),
                         corr(volatile_acidity, density),
                         corr(volatile_acidity, pH),
                         corr(volatile_acidity, sulphates),
                         corr(volatile_acidity, alcohol)];

subplot(3,4,2);
bar (corr_volatile_acidity);
xlabel("Attributes");
ylabel("Correlation");
title("Volatile Acidity");

corr_citric_acid = [corr(citric_acid, fixed_acidity),
                    corr(citric_acid, volatile_acidity),
                    corr(citric_acid, citric_acid),
                    corr(citric_acid, residual_sugar),
                    corr(citric_acid, chlorides),
                    corr(citric_acid, free_sulfur_dioxide),
                    corr(citric_acid, total_sulfur_dioxide),
                    corr(citric_acid, density),
                    corr(citric_acid, pH),
                    corr(citric_acid, sulphates),
                    corr(citric_acid, alcohol)];

subplot(3,4,3);
bar (corr_citric_acid);
xlabel("Attributes");
ylabel("Correlation");
title("Citric Acid");

corr_residual_sugar = [corr(residual_sugar, fixed_acidity),
                       corr(residual_sugar, volatile_acidity),
                       corr(residual_sugar, citric_acid),
                       corr(residual_sugar, residual_sugar),
                       corr(residual_sugar, chlorides),
                       corr(residual_sugar, free_sulfur_dioxide),
                       corr(residual_sugar, total_sulfur_dioxide),
                       corr(residual_sugar, density),
                       corr(residual_sugar, pH),
                       corr(residual_sugar, sulphates),
                       corr(residual_sugar, alcohol)];

subplot(3,4,4);
bar (corr_residual_sugar);
xlabel("Attributes");
ylabel("Correlation");
title("Residual Sugar");

corr_chlorides = [corr(chlorides, fixed_acidity),
                  corr(chlorides, volatile_acidity),
                  corr(chlorides, citric_acid),
                  corr(chlorides, residual_sugar),
                  corr(chlorides, chlorides),
                  corr(chlorides, free_sulfur_dioxide),
                  corr(chlorides, total_sulfur_dioxide),
                  corr(chlorides, density),
                  corr(chlorides, pH),
                  corr(chlorides, sulphates),
                  corr(chlorides, alcohol)];

subplot(3,4,5);
bar (corr_chlorides);
xlabel("Attributes");
ylabel("Correlation");
title("Chlorides");

corr_free_sulfur_dioxide = [corr(free_sulfur_dioxide, fixed_acidity),
                            corr(free_sulfur_dioxide, volatile_acidity),
                            corr(free_sulfur_dioxide, citric_acid),
                            corr(free_sulfur_dioxide, residual_sugar),
                            corr(free_sulfur_dioxide, chlorides),
                            corr(free_sulfur_dioxide, free_sulfur_dioxide),
                            corr(free_sulfur_dioxide, total_sulfur_dioxide),
                            corr(free_sulfur_dioxide, density),
                            corr(free_sulfur_dioxide, pH),
                            corr(free_sulfur_dioxide, sulphates),
                            corr(free_sulfur_dioxide, alcohol)];

subplot(3,4,6);
bar (corr_free_sulfur_dioxide);
xlabel("Attributes");
ylabel("Correlation");
title("Free Sulfur Dioxide");

corr_total_sulfur_dioxide = [corr(total_sulfur_dioxide, fixed_acidity),
                            corr(total_sulfur_dioxide, volatile_acidity),
                            corr(total_sulfur_dioxide, citric_acid),
                            corr(total_sulfur_dioxide, residual_sugar),
                            corr(total_sulfur_dioxide, chlorides),
                            corr(total_sulfur_dioxide, free_sulfur_dioxide),
                            corr(total_sulfur_dioxide, total_sulfur_dioxide),
                            corr(total_sulfur_dioxide, density),
                            corr(total_sulfur_dioxide, pH),
                            corr(total_sulfur_dioxide, sulphates),
                            corr(total_sulfur_dioxide, alcohol)];

subplot(3,4,7);
bar (corr_total_sulfur_dioxide);
xlabel("Attributes");
ylabel("Correlation");
title("Total Sulfur Dioxide");

corr_density = [corr(density, fixed_acidity),
                corr(density, volatile_acidity),
                corr(density, citric_acid),
                corr(density, residual_sugar),
                corr(density, chlorides),
                corr(density, free_sulfur_dioxide),
                corr(density, total_sulfur_dioxide),
                corr(density, density),
                corr(density, pH),
                corr(density, sulphates),
                corr(density, alcohol)];

subplot(3,4,8);
bar (corr_density);
xlabel("Attributes");
ylabel("Correlation");
title("Density");

corr_pH = [corr(pH, fixed_acidity),
           corr(pH, volatile_acidity),
           corr(pH, citric_acid),
           corr(pH, residual_sugar),
           corr(pH, chlorides),
           corr(pH, free_sulfur_dioxide),
           corr(pH, total_sulfur_dioxide),
           corr(pH, density),
           corr(pH, pH),
           corr(pH, sulphates),
           corr(pH, alcohol)];

subplot(3,4,9);
bar (corr_pH);
xlabel("Attributes");
ylabel("Correlation");
title("pH");

corr_sulphates = [corr(sulphates, fixed_acidity),
                  corr(sulphates, volatile_acidity),
                  corr(sulphates, citric_acid),
                  corr(sulphates, residual_sugar),
                  corr(sulphates, chlorides),
                  corr(sulphates, free_sulfur_dioxide),
                  corr(sulphates, total_sulfur_dioxide),
                  corr(sulphates, density),
                  corr(sulphates, pH),
                  corr(sulphates, sulphates),
                  corr(sulphates, alcohol)];

subplot(3,4,10);
bar (corr_sulphates);
xlabel("Attributes");
ylabel("Correlation");
title("Sulphates");

corr_alcohol = [corr(alcohol, fixed_acidity),
                corr(alcohol, volatile_acidity),
                corr(alcohol, citric_acid),
                corr(alcohol, residual_sugar),
                corr(alcohol, chlorides),
                corr(alcohol, free_sulfur_dioxide),
                corr(alcohol, total_sulfur_dioxide),
                corr(alcohol, density),
                corr(alcohol, pH),
                corr(alcohol, sulphates),
                corr(alcohol, alcohol)];

subplot(3,4,11);
bar (corr_alcohol);
xlabel("Attributes");
ylabel("Correlation");
title("Alcohol");

# From the graphs we can see that density has high correlation with two other
# attributes (residual sugar and alcohol), perhaps removing it will help improve
# the overall accuracy...

################################################################################
# Evaluating training data with removed attribute Density
class_probabilities = [0;0;0;0;0;0;0;0;0];
likelihoods         = [0;0;0;0;0;0;0;0;0];
probabilities_fixed_acidity        = [0;0;0;0;0;0;0;0;0];
probabilities_volatile_acidity     = [0;0;0;0;0;0;0;0;0];
probabilities_citric_acid          = [0;0;0;0;0;0;0;0;0];
probabilities_residual_sugar       = [0;0;0;0;0;0;0;0;0];
probabilities_chlorides            = [0;0;0;0;0;0;0;0;0];
probabilities_free_sulfur_dioxide  = [0;0;0;0;0;0;0;0;0];
probabilities_total_sulfur_dioxide = [0;0;0;0;0;0;0;0;0];
probabilities_pH                   = [0;0;0;0;0;0;0;0;0];
probabilities_sulphates            = [0;0;0;0;0;0;0;0;0];
probabilities_alcohol              = [0;0;0;0;0;0;0;0;0];

guesses  = 0;
correct  = 0;
accuracy = 0;

disp("Evaluating training set with removed attribute Density:\n______________________");
for i = 1:rows(wine_data)
  for j = 1:rows(priors)
    probabilities_fixed_acidity(j) = (1 / sqrt((2*pi*sigma_fixed_acidity(j)))) * exp (-0.5*((fixed_acidity(i) - mu_fixed_acidity(j))^2 / sigma_fixed_acidity(j)));
    if (probabilities_fixed_acidity(j)==0) probabilities_fixed_acidity(j) = realmin; endif
    
    probabilities_volatile_acidity(j) = (1 / sqrt((2*pi*sigma_volatile_acidity(j)))) * exp (-0.5*((volatile_acidity(i) - mu_volatile_acidity(j))^2 / sigma_volatile_acidity(j)));
    if (probabilities_volatile_acidity(j)==0) probabilities_volatile_acidity(j) = realmin; endif
    
    probabilities_citric_acid(j) = (1 / sqrt((2*pi*sigma_citric_acid(j)))) * exp (-0.5*((citric_acid(i) - mu_citric_acid(j))^2 / sigma_citric_acid(j)));
    if (probabilities_citric_acid(j)==0) probabilities_citric_acid(j) = realmin; endif
    
    probabilities_residual_sugar(j) = (1 / sqrt((2*pi*sigma_residual_sugar(j)))) * exp (-0.5*((residual_sugar(i) - mu_residual_sugar(j))^2 / sigma_residual_sugar(j)));
    if (probabilities_residual_sugar(j)==0) probabilities_residual_sugar(j) = realmin; endif
    
    probabilities_chlorides(j) = (1 / sqrt((2*pi*sigma_chlorides(j)))) * exp (-0.5*((chlorides(i) - mu_chlorides(j))^2 / sigma_chlorides(j)));
    if (probabilities_chlorides(j)==0) probabilities_chlorides(j) = realmin; endif
    
    probabilities_free_sulfur_dioxide(j) = (1 / sqrt((2*pi*sigma_free_sulfur_dioxide(j)))) * exp (-0.5*((free_sulfur_dioxide(i) - mu_free_sulfur_dioxide(j))^2 / sigma_free_sulfur_dioxide(j)));
    if (probabilities_free_sulfur_dioxide(j)==0) probabilities_free_sulfur_dioxide(j) = realmin; endif
    
    probabilities_total_sulfur_dioxide(j) = (1 / sqrt((2*pi*sigma_total_sulfur_dioxide(j)))) * exp (-0.5*((total_sulfur_dioxide(i) - mu_total_sulfur_dioxide(j))^2 / sigma_total_sulfur_dioxide(j)));
    if (probabilities_total_sulfur_dioxide(j)==0) probabilities_total_sulfur_dioxide(j) = realmin; endif
    
    probabilities_pH(j) = (1 / sqrt((2*pi*sigma_pH(j)))) * exp (-0.5*((pH(i) - mu_pH(j))^2 / sigma_pH(j)));
    if (probabilities_pH(j)==0) probabilities_pH(j) = realmin; endif
    
    probabilities_sulphates(j) = (1 / sqrt((2*pi*sigma_sulphates(j)))) * exp (-0.5*((sulphates(i) - mu_sulphates(j))^2 / sigma_sulphates(j)));
    if (probabilities_sulphates(j)==0) probabilities_sulphates(j) = realmin; endif
    
    probabilities_alcohol(j) = (1 / sqrt((2*pi*sigma_alcohol(j)))) * exp (-0.5*((alcohol(i) - mu_alcohol(j))^2 / sigma_alcohol(j)));
    if (probabilities_alcohol(j)==0) probabilities_alcohol(j) = realmin; endif
  endfor
  
  p_x_given_a = [0;0;0;0;0;0;0;0;0];
  for p = 1:rows(p_x_given_a)
    p_x_given_a(p) = probabilities_fixed_acidity(p) * probabilities_volatile_acidity(p) * probabilities_citric_acid(p) * probabilities_residual_sugar(p) * probabilities_chlorides(p) * probabilities_free_sulfur_dioxide(p) * probabilities_total_sulfur_dioxide(p) * probabilities_pH(p) * probabilities_sulphates(p) * probabilities_alcohol(p);
  endfor
  
  for k = 1:rows(likelihoods)
    likelihoods(k) = priors(k) * p_x_given_a(k);
  endfor
  
  prediction = 0;
  highest_probability = 0;
  
  for n = 1:rows(class_probabilities)
    class_probabilities(n) = likelihoods(n) / sum (likelihoods);
    y_sub_k = class_probabilities(n);
    if (y_sub_k > highest_probability)
      highest_probability = y_sub_k;
      prediction = n;
    endif
  endfor
  
  if (prediction == quality(i))
    correct = correct + 1;
    guesses = guesses + 1;
  else
    guesses = guesses + 1;
  endif
  
  accuracy = correct / guesses * 100;
  
  #printf("Prediction: %i, Actual: %i\nAccuracy = %f\n_______________________\n", prediction, quality(i), accuracy);
endfor

printf("Overall new Naive Bayes Model accuracy: %f\n", accuracy);
disp("");

################################################################################
# Evaluating the model with the Test Data
# Import the test data
test_data = csvread ("./WineData/wine_test_set.csv",1,0);

fixed_acidity        = test_data(:,1);
volatile_acidity     = test_data(:,2);
citric_acid          = test_data(:,3);
residual_sugar       = test_data(:,4);
chlorides            = test_data(:,5);
free_sulfur_dioxide  = test_data(:,6);
total_sulfur_dioxide = test_data(:,7);
density              = test_data(:,8);
pH                   = test_data(:,9);
sulphates            = test_data(:,10);
alcohol              = test_data(:,11);
quality              = test_data(:,12);

# Calculate probability evidence
# Zero out the training results
# Could this be made into a function? Yes
# Do i care to do that? No, nor do i care to learn how
class_probabilities = [0;0;0;0;0;0;0;0;0];
likelihoods         = [0;0;0;0;0;0;0;0;0];
probabilities_fixed_acidity        = [0;0;0;0;0;0;0;0;0];
probabilities_volatile_acidity     = [0;0;0;0;0;0;0;0;0];
probabilities_citric_acid          = [0;0;0;0;0;0;0;0;0];
probabilities_residual_sugar       = [0;0;0;0;0;0;0;0;0];
probabilities_chlorides            = [0;0;0;0;0;0;0;0;0];
probabilities_free_sulfur_dioxide  = [0;0;0;0;0;0;0;0;0];
probabilities_total_sulfur_dioxide = [0;0;0;0;0;0;0;0;0];
probabilities_density              = [0;0;0;0;0;0;0;0;0];
probabilities_pH                   = [0;0;0;0;0;0;0;0;0];
probabilities_sulphates            = [0;0;0;0;0;0;0;0;0];
probabilities_alcohol              = [0;0;0;0;0;0;0;0;0];

guesses  = 0;
correct  = 0;
accuracy = 0;

disp("Evaluating test set:\n______________________");
for i = 1:rows(test_data)
  for j = 1:rows(priors)
    probabilities_fixed_acidity(j) = (1 / sqrt((2*pi*sigma_fixed_acidity(j)))) * exp (-0.5*((fixed_acidity(i) - mu_fixed_acidity(j))^2 / sigma_fixed_acidity(j)));
    if (probabilities_fixed_acidity(j)==0) probabilities_fixed_acidity(j) = realmin; endif
    
    probabilities_volatile_acidity(j) = (1 / sqrt((2*pi*sigma_volatile_acidity(j)))) * exp (-0.5*((volatile_acidity(i) - mu_volatile_acidity(j))^2 / sigma_volatile_acidity(j)));
    if (probabilities_volatile_acidity(j)==0) probabilities_volatile_acidity(j) = realmin; endif
    
    probabilities_citric_acid(j) = (1 / sqrt((2*pi*sigma_citric_acid(j)))) * exp (-0.5*((citric_acid(i) - mu_citric_acid(j))^2 / sigma_citric_acid(j)));
    if (probabilities_citric_acid(j)==0) probabilities_citric_acid(j) = realmin; endif
    
    probabilities_residual_sugar(j) = (1 / sqrt((2*pi*sigma_residual_sugar(j)))) * exp (-0.5*((residual_sugar(i) - mu_residual_sugar(j))^2 / sigma_residual_sugar(j)));
    if (probabilities_residual_sugar(j)==0) probabilities_residual_sugar(j) = realmin; endif
    
    probabilities_chlorides(j) = (1 / sqrt((2*pi*sigma_chlorides(j)))) * exp (-0.5*((chlorides(i) - mu_chlorides(j))^2 / sigma_chlorides(j)));
    if (probabilities_chlorides(j)==0) probabilities_chlorides(j) = realmin; endif
    
    probabilities_free_sulfur_dioxide(j) = (1 / sqrt((2*pi*sigma_free_sulfur_dioxide(j)))) * exp (-0.5*((free_sulfur_dioxide(i) - mu_free_sulfur_dioxide(j))^2 / sigma_free_sulfur_dioxide(j)));
    if (probabilities_free_sulfur_dioxide(j)==0) probabilities_free_sulfur_dioxide(j) = realmin; endif
    
    probabilities_total_sulfur_dioxide(j) = (1 / sqrt((2*pi*sigma_total_sulfur_dioxide(j)))) * exp (-0.5*((total_sulfur_dioxide(i) - mu_total_sulfur_dioxide(j))^2 / sigma_total_sulfur_dioxide(j)));
    if (probabilities_total_sulfur_dioxide(j)==0) probabilities_total_sulfur_dioxide(j) = realmin; endif
    
    probabilities_density(j) = (1 / sqrt((2*pi*sigma_density(j)))) * exp (-0.5*((density(i) - mu_density(j))^2 / sigma_density(j)));
    if (probabilities_density(j)==0) probabilities_density(j) = realmin; endif
    
    probabilities_pH(j) = (1 / sqrt((2*pi*sigma_pH(j)))) * exp (-0.5*((pH(i) - mu_pH(j))^2 / sigma_pH(j)));
    if (probabilities_pH(j)==0) probabilities_pH(j) = realmin; endif
    
    probabilities_sulphates(j) = (1 / sqrt((2*pi*sigma_sulphates(j)))) * exp (-0.5*((sulphates(i) - mu_sulphates(j))^2 / sigma_sulphates(j)));
    if (probabilities_sulphates(j)==0) probabilities_sulphates(j) = realmin; endif
    
    probabilities_alcohol(j) = (1 / sqrt((2*pi*sigma_alcohol(j)))) * exp (-0.5*((alcohol(i) - mu_alcohol(j))^2 / sigma_alcohol(j)));
    if (probabilities_alcohol(j)==0) probabilities_alcohol(j) = realmin; endif
  endfor
  
  p_x_given_a = [0;0;0;0;0;0;0;0;0];
  for p = 1:rows(p_x_given_a)
    p_x_given_a(p) = probabilities_fixed_acidity(p) * probabilities_volatile_acidity(p) * probabilities_citric_acid(p) * probabilities_residual_sugar(p) * probabilities_chlorides(p) * probabilities_free_sulfur_dioxide(p) * probabilities_total_sulfur_dioxide(p) * probabilities_density(p) * probabilities_pH(p) * probabilities_sulphates(p) * probabilities_alcohol(p);
  endfor
  
  for k = 1:rows(likelihoods)
    likelihoods(k) = priors(k) * p_x_given_a(k);
  endfor
  
  prediction = 0;
  highest_probability = 0;
  
  for n = 1:rows(class_probabilities)
    class_probabilities(n) = likelihoods(n) / sum (likelihoods);
    y_sub_k = class_probabilities(n);
    if (y_sub_k > highest_probability)
      highest_probability = y_sub_k;
      prediction = n;
    endif
  endfor
  
  if (prediction == quality(i))
    correct = correct + 1;
    guesses = guesses + 1;
  else
    guesses = guesses + 1;
  endif
  
  accuracy = correct / guesses * 100;
  
  #printf("Prediction: %i, Actual: %i\nAccuracy = %f\n_______________________\n", prediction, quality(i), accuracy);
endfor

printf("Overall Naive Bayes Model accuracy: %f\n", accuracy);
disp("");

################################################################################
# Evaluating test data with removed attribute Density
class_probabilities = [0;0;0;0;0;0;0;0;0];
likelihoods         = [0;0;0;0;0;0;0;0;0];
probabilities_fixed_acidity        = [0;0;0;0;0;0;0;0;0];
probabilities_volatile_acidity     = [0;0;0;0;0;0;0;0;0];
probabilities_citric_acid          = [0;0;0;0;0;0;0;0;0];
probabilities_residual_sugar       = [0;0;0;0;0;0;0;0;0];
probabilities_chlorides            = [0;0;0;0;0;0;0;0;0];
probabilities_free_sulfur_dioxide  = [0;0;0;0;0;0;0;0;0];
probabilities_total_sulfur_dioxide = [0;0;0;0;0;0;0;0;0];
probabilities_pH                   = [0;0;0;0;0;0;0;0;0];
probabilities_sulphates            = [0;0;0;0;0;0;0;0;0];
probabilities_alcohol              = [0;0;0;0;0;0;0;0;0];

guesses  = 0;
correct  = 0;
accuracy = 0;

disp("Evaluating test set with removed attribute Density:\n______________________");
for i = 1:rows(test_data)
  for j = 1:rows(priors)
    probabilities_fixed_acidity(j) = (1 / sqrt((2*pi*sigma_fixed_acidity(j)))) * exp (-0.5*((fixed_acidity(i) - mu_fixed_acidity(j))^2 / sigma_fixed_acidity(j)));
    if (probabilities_fixed_acidity(j)==0) probabilities_fixed_acidity(j) = realmin; endif
    
    probabilities_volatile_acidity(j) = (1 / sqrt((2*pi*sigma_volatile_acidity(j)))) * exp (-0.5*((volatile_acidity(i) - mu_volatile_acidity(j))^2 / sigma_volatile_acidity(j)));
    if (probabilities_volatile_acidity(j)==0) probabilities_volatile_acidity(j) = realmin; endif
    
    probabilities_citric_acid(j) = (1 / sqrt((2*pi*sigma_citric_acid(j)))) * exp (-0.5*((citric_acid(i) - mu_citric_acid(j))^2 / sigma_citric_acid(j)));
    if (probabilities_citric_acid(j)==0) probabilities_citric_acid(j) = realmin; endif
    
    probabilities_residual_sugar(j) = (1 / sqrt((2*pi*sigma_residual_sugar(j)))) * exp (-0.5*((residual_sugar(i) - mu_residual_sugar(j))^2 / sigma_residual_sugar(j)));
    if (probabilities_residual_sugar(j)==0) probabilities_residual_sugar(j) = realmin; endif
    
    probabilities_chlorides(j) = (1 / sqrt((2*pi*sigma_chlorides(j)))) * exp (-0.5*((chlorides(i) - mu_chlorides(j))^2 / sigma_chlorides(j)));
    if (probabilities_chlorides(j)==0) probabilities_chlorides(j) = realmin; endif
    
    probabilities_free_sulfur_dioxide(j) = (1 / sqrt((2*pi*sigma_free_sulfur_dioxide(j)))) * exp (-0.5*((free_sulfur_dioxide(i) - mu_free_sulfur_dioxide(j))^2 / sigma_free_sulfur_dioxide(j)));
    if (probabilities_free_sulfur_dioxide(j)==0) probabilities_free_sulfur_dioxide(j) = realmin; endif
    
    probabilities_total_sulfur_dioxide(j) = (1 / sqrt((2*pi*sigma_total_sulfur_dioxide(j)))) * exp (-0.5*((total_sulfur_dioxide(i) - mu_total_sulfur_dioxide(j))^2 / sigma_total_sulfur_dioxide(j)));
    if (probabilities_total_sulfur_dioxide(j)==0) probabilities_total_sulfur_dioxide(j) = realmin; endif
    
    probabilities_pH(j) = (1 / sqrt((2*pi*sigma_pH(j)))) * exp (-0.5*((pH(i) - mu_pH(j))^2 / sigma_pH(j)));
    if (probabilities_pH(j)==0) probabilities_pH(j) = realmin; endif
    
    probabilities_sulphates(j) = (1 / sqrt((2*pi*sigma_sulphates(j)))) * exp (-0.5*((sulphates(i) - mu_sulphates(j))^2 / sigma_sulphates(j)));
    if (probabilities_sulphates(j)==0) probabilities_sulphates(j) = realmin; endif
    
    probabilities_alcohol(j) = (1 / sqrt((2*pi*sigma_alcohol(j)))) * exp (-0.5*((alcohol(i) - mu_alcohol(j))^2 / sigma_alcohol(j)));
    if (probabilities_alcohol(j)==0) probabilities_alcohol(j) = realmin; endif
  endfor
  
  p_x_given_a = [0;0;0;0;0;0;0;0;0];
  for p = 1:rows(p_x_given_a)
    p_x_given_a(p) = probabilities_fixed_acidity(p) * probabilities_volatile_acidity(p) * probabilities_citric_acid(p) * probabilities_residual_sugar(p) * probabilities_chlorides(p) * probabilities_free_sulfur_dioxide(p) * probabilities_total_sulfur_dioxide(p) * probabilities_pH(p) * probabilities_sulphates(p) * probabilities_alcohol(p);
  endfor
  
  for k = 1:rows(likelihoods)
    likelihoods(k) = priors(k) * p_x_given_a(k);
  endfor
  
  prediction = 0;
  highest_probability = 0;
  
  for n = 1:rows(class_probabilities)
    class_probabilities(n) = likelihoods(n) / sum (likelihoods);
    y_sub_k = class_probabilities(n);
    if (y_sub_k > highest_probability)
      highest_probability = y_sub_k;
      prediction = n;
    endif
  endfor
  
  if (prediction == quality(i))
    correct = correct + 1;
    guesses = guesses + 1;
  else
    guesses = guesses + 1;
  endif
  
  accuracy = correct / guesses * 100;
  
  #printf("Prediction: %i, Actual: %i\nAccuracy = %f\n_______________________\n", prediction, quality(i), accuracy);
endfor

printf("Overall new Naive Bayes Model accuracy: %f\n", accuracy);
disp("");