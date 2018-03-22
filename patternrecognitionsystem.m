function patternRecognitionSystem(X_features, X_classes, k)
    % Pattern recognition system
    % OUTPUTS:
    % INPUTS:
    % X_features       matrix containing the data
    % X_classes        correct labels for the data

    % Do the following first:
    %
    % % Clear and close all
    % close all;
    % clear;
    % clc;
    %
    % % Load data
    %load ex7_data.mat
    %
    %X_features = X(:, 2:end);
    %X_classes = X(:,1);
    %
    %k = 1; % 1-NN

    N = size(X_features,1);
    num_features = size(X_features,2);

    % Whitening the data
    X_stand = standardize(X_features);
    %X_stand=X_features;
    % Select training and validation sets
    % Forward search implements leave-one-out cross validation therefore a separate test set is
    % included in training
    % 2/3 for training and 1/3 for validation
    rng(7) % to see how changes we made changed the validation_result we fix the 'seed' to get the same random ordering of number
    selection = randperm(N); % see also 'help rng'
    training_data = X_stand(selection(1:floor(2*N/3)), :);
    validation_data = X_stand(selection((floor(2*N/3)+1):N), :);
    training_class = X_classes(selection(1:floor(2*N/3)), 1);
    validation_class = X_classes(selection((floor(2*N/3)+1):N), 1);

    % Train feature vector
    fvector = zeros(num_features,1);
    best_result = 0;
    
    for in = 1:num_features
        [best_result_add, best_feature_add] = forwardsearch(training_data, training_class, fvector, k);
        % Update the feature vector
        fvector(best_feature_add) = 1;

        % Save best result
        if(best_result < best_result_add)
            best_result = best_result_add;
            best_fvector = fvector;
        end

    end

    best_result
    best_fvector

    % Test
    valid_res = knnclass(validation_data, training_data, best_fvector, training_class, k); % valid_res equals predicted class for validation_data
    correct = sum(valid_res == validation_class); % amount of correct samples
    validation_result = correct/length(validation_class)

    %[CMAT, ORDER] = confusionmat(validation_class, valid_res)

end

function [feat_out] = standardize(feat_in)
    N = size(feat_in,1); 
    % centering
    feat_cent = feat_in-repmat(mean(feat_in), N, 1);
    % standardization
    feat_stand = feat_cent./repmat(std(feat_cent), N, 1);

    % whitening eigenvalue decomposition
    [V,D] = eig(cov(feat_cent)); %see help eig
    W = sqrt(inv(D)) * V' ;
    z=W* feat_cent'; % Matlab cov() takes sample size into account and scaling is therefore not required

    feat_whit2 = z';
    %cov(feat_whit2)


    % method 3 whitening using  SVD  (inv(S) = diag(1./diag(S) for diagonal matrices if speed is desired)


    % or without cov
    %[U,S,V] = svd(feat_cent,0);
    %Y = inv(S/sqrt(N-1))*V'*feat_cent'; % Notice the sample size scaling and that singular values S are sqrt of eigenvalues

    feat_out = feat_whit2; % choose whitening method by hand
    %feat_out=feat_stand;

end

function [predictedLabels] = knnclass(dat1, dat2, fvec, classes, k)
    p1 = pdist2( dat1(:,logical(fvec)), dat2(:,logical(fvec)) );
    % Here we aim in finding k-smallest elements
    [D, I] = sort(p1', 1);
    I = I(1:k+1, :);
    labels = classes( : )';
    if k == 1 % this is for k-NN, k = 1
        predictedLabels = labels( I(2, : ) )';
    else % this is for k-NN, other odd k larger than 1
        predictedLabels = mode( labels( I( 1+(1:k), : ) ), 1)'; % see help mode
    end
end

function [best, feature] = forwardsearch(data, data_c, fvector, k)
    % SFS, from previous lesson.
    num_samples = length(data);
    best = 0;
    feature = 0;


    for in = 1:length(fvector)
        if (fvector(in) == 0)
            fvector(in) = 1;
            % Classify using k-NN
            predictedLabels = knnclass(data, data, fvector, data_c, k);
            correct = sum(predictedLabels == data_c); % the number of correct predictions
            result = correct/num_samples; % accuracy
            if(result > best)
                best = result;
                feature = in;
            end
            fvector(in) = 0;
        end
    end
end