%% TITLE ****************************************************************
% *                                                                      *
% *              		 521289S Machine Learning 					     *
% *                     Programming Assignment 2018                      *
% *                                                                      *
% *   Author 1: << Insert Name and Student ID number here >>             *
% *                                                                      *
% *   NOTE: The file name for this file MUST BE 'classify.m'!            *
% *         Everything should be included in this single file.           *
% *                                                                      *
% ************************************************************************

%% NOTE ******************************************************************
% *                                                                      *
% *       DO NOT DEFINE ANY GLOBAL VARIABLES (outside functions)!        *
% *                                                                      *
% *       Your task is to complete the PUBLIC INTERFACE below without    *
% *       modifying the definitions. You can define and implement any    *
% *       functions you like in the PRIVATE INTERFACE at the end of      *
% *       this file as you wish.                                         *
% *                                                                      *
% ************************************************************************

%% HINT ******************************************************************
% *                                                                      *
% *       If you enable cell folding for the m-file editor, you can      *
% *       easily hide these comments from view for less clutter!         *
% *                                                                      *
% *       [ File -> Preferences -> Code folding -> Cells = enable.       *
% *         Then use -/+ signs on the left margin for hiding/showing. ]  *
% *                                                                      *
% ************************************************************************

%% This is the main function of this m-file
%  You can use this e.g. for unit testing.
%
% INPUT:  none (change if you wish)
%
% OUTPUT: none (change if you wish)
%%
function classify()
% Typically, you could: 
% - Load the data.
% - Split the data to training and validation sets.
% - Train the classifier on the training set (call trainClassifier).
% - Test it using the validation data set and learned parameters (call
%   evaluateClassifier).
% - Calculate performance statistics (accuracy, sensitivity, specificity,
%   etc.)
    
    dataSize = 0; % if 0 use whole data set
    
    training_data_file = 'trainingdata.mat';
    load(training_data_file, 'trainingData'); 
    load(training_data_file, 'class_trainingData');

    if dataSize && dataSize < size(trainingData,1) %#ok<NODEF>
        learn_data = trainingData(1:(floor(2/3*dataSize)),:);
        learn_classes = class_trainingData(1:(floor(2/3*dataSize)));
        test_data = trainingData((floor(2/3*dataSize))+1:dataSize,:);
        test_classes = class_trainingData((floor(2/3*dataSize))+1:dataSize);
    else
        learn_data = trainingData(1:5500,:);
        learn_classes = class_trainingData(1:5500);
        test_data = trainingData(5501:end,:);
        test_classes = class_trainingData(5501:end); %#ok<COLND>
    end
    clear trainingData;
    clear class_trainingData;

    [learnDataAmount, ~] = size(learn_data);
    [learnClassAmount, ~] = size(learn_classes);
    if learnDataAmount == learnClassAmount
        display('learn size matches');
    end
    
    [testDataAmount, ~] = size(test_data);
    [testClassAmount, ~] = size(test_classes);
    if testDataAmount == testClassAmount
        display('test size matches');
    end
    
    parameters = trainClassifier(learn_data, learn_classes);
    
  
    correct = 0;
    results = evaluateClassifier(test_data, parameters);

    for i = 1:size(results, 1) 
        winnerclass = results(i);
        realWinner = test_classes(i);
        if realWinner == winnerclass
            correct = correct + 1;
        end
    end
    correctnessRate = correct / testDataAmount;
    display(['Correctness rate ', num2str(correctnessRate)]);
            
end


%% PUBLIC INTERFACE ******************************************************
% *                                                                      *
% *   Below are the functions that define the classifier training and    *
% *   evaluation. Your task is to complete these!                        *
% *                                                                      *
% *   NOTE: You MUST NOT change the function definitions that describe   *
% *         the input and output variables, and the names of the         *
% *         functions! Otherwise, the automatic ranking system cannot    *
% *         evaluate your algorithm!                                     *
% *                                                                      *
% ************************************************************************


%% This function gives the nick name that is shown in the ranking list
% at the course web page. Use 1-15 characters (a-z, A-Z, 0-9 or _).
%
% Check the rankings page beforehand to guarantee an unique nickname:
% http://www.ee.oulu.fi/research/tklab/courses/521289S/progex/rankings.html
% 
% INPUT:  none
%
% OUTPUT: Please change this to be a unique name and do not alter it 
% if resubmitting a new version to the ranking system for re-evaluation!
%%
function nick = getNickName() %#ok<DEFNU>
    nick = 'BOT_Zed';
end


%% This is the training interface for the classifier you are constructing.
%  All the learning takes place here.
%
% INPUT:  
%
%   samples:
%            A N-by-M data matrix. The rows represent samples and 
%            the columns features. N is the number of samples.
%            M is the number of features.
%
%            This could be e.g. the training data matrix given on the 
%            course web page or the validation data set that has been 
%            withheld for the validation on the server side.
%
%            Note: The value for N can vary! Do not hard-code it!
%
%   classes:
%
%            A N-by-1 vector of correct classes. Each row gives the correct
%            class for the corresponding data in the samples matrix.
%
% OUTPUT: 
%
%   parameters:
%            Any type of data structure supported by MATLAB. You decide!
%            You should use this to store the results of the training.
%
%            This set of parameters is given to the classifying function
%            that can operate on a completely different set of data.
%
%            For example, a classifier based on discriminant functions
%            could store here the weight vectors/matrices that define 
%            the functions. A kNN-classifier would store all the training  
%            data samples, their classification, and the value chosen 
%            for the k-parameter.
%            
%            Especially, structure arrays (keyword: struct) are useful for 
%            storing multiple parameters of different type in a single 
%            struct. Cell arrays could also be useful.
%            See MATLAB help for details on these.
%%
function parameters = trainClassifier( samples, classes )
%%
% Insert the function body here!
%
% You must be able to construct a classifier solely based on the data in
% the parameters data structure in the evaluateClassifier-function.
% Consequently, choose carefully what is needed to be stored in the
% parameters data structure.
%
% Hint 1: If you wish to avoid overtraining, you could partition the
% training samples to a actual training data set and a testing data set
% that is used for a stopping/overtraining criterion.
%
% Hint 2: To avoid duplicating e.g. classifier construction code define
% suitable own functions at the end of this file as subfunctions!
% You could also utilize nested functions within this function
% to perform repetitive tasks. See MATLAB help for details.
%
% You are free to remove these comments.
%
    max_k = 5;
    %samples = standardize(samples);
    [~, num_features] = size(samples);
    
	% Train feature vector
    fvector = zeros(num_features,1);
    best_result = 0;
	for k = 1:max_k
        t_fvector = zeros(num_features,1);
        t_best_result = 0;
        for in = 1:num_features
            [best_result_add, best_feature_add] = forwardsearch(samples, classes, t_fvector, k);
            % Update the feature vector
            t_fvector(best_feature_add) = 1;

            % Save best result
            if(t_best_result < best_result_add)
                t_best_result = best_result_add;
                t_best_fvector = t_fvector;
                if(best_result < best_result_add)
                    best_result = best_result_add;
                    best_fvector = t_fvector;
                    best_k = k;
                end
            end
        end
	end
    parameters.trainingData = samples;
    parameters.trainingClasses = classes;
    parameters.k = best_k;
    parameters.best_fvector = best_fvector;
end


%% This is the evaluation interface of your classifier.
%  This function is used to perform the actual classification of a set of
%  samples given a fixed set of parameters defining the classifier.
%
% INPUT:   
%   samples:
%            A N-by-M data matrix. The rows represent samples and 
%            the columns features. N is the number of samples.
%            M is the number of features.
%
%            Note that N could be different from what it was in the
%            previous training function!
%
%   parameters:
%            Any type of data structure supported by MATLAB.
%
%            This is the output of the trainClassifier function you have
%            implemented above.
%
% OUTPUT: 
%   results:
%            The results of the classification as a N-by-1 vector of
%            estimated classes.
%
%            The data type and value range must correspond to the classes
%            vector in the previous function.
%%
function results = evaluateClassifier( samples, parameters )
    %%
    % Insert the function body here!
    %
    % Typically, you must construct the classifier with the given parameters
    % and then apply it to the given data.
    %
    % Note that no learning should take place in this function! The classifier
    % is just applied to the data.
    %
    % Hint: To avoid duplicating e.g. classifier construction code define
    % suitable own functions at the end of this file as subfunctions!
    % You could also utilize nested functions within this function
    % to perform repetitive tasks. See MATLAB help for details.
    %
    % You are free to remove these comments.
    %
    %mySom = parameters;
    %samples = standardize(samples);
    [~, ~] = size(samples);
    
    training_class = parameters.trainingClasses;
    training_samples = parameters.trainingData;
    best_fvector = parameters.best_fvector;
    k = parameters.k;
    %results = zeros(N, 1);
    results = knnclass(samples, training_samples, best_fvector, training_class, k);
    %display(results);
end


%% PRIVATE INTERFACE *****************************************************
% *                                                                      *
% *   User defined functions that are needed e.g. for training and       *
% *   evaluating the classifier above.                                   *
% *                                                                      *
% *   Please note that these are subfunctions that are visible only to   *
% *   the other functions in this file. These are defined using the      *
% *   'function' keyword after the body of the preceding functions or    *
% *   subfunctions. Subfunctions are not visible outside the file where  *
% *   they are defined.                                                  *
% *                                                                      *
% *   To avoid calling MATLAB toolbox functions that are not available   *
% *   on the server side, implement those here.                          *
% *                                                                      *
% ************************************************************************

%% A simple example:
%  (You can delete this if you wish!)
%{
function d = myDistanceFunction( x, y )
    d = sqrt( sum( (x - y).^2 ) );
end

% From ex7
function feat_in = standardize(feat_in)
	N = size(feat_in,1); 
	% centering
	feat_cent = feat_in-repmat(mean(feat_in), N, 1);
	% standardization
	% feat_stand = feat_cent./repmat(std(feat_cent), N, 1);

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

	feat_in = feat_whit2; % choose whitening method by hand
	%feat_out=feat_stand;
end
%}
function predictedLabels = knnclass(dat1, dat2, fvec, classes, k)
	p1 = pdist2( dat1(:,logical(fvec)), dat2(:,logical(fvec)) );
	% Here we aim in finding k-smallest elements
	[~, I] = sort(p1', 1);
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
   