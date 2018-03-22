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
    %{
    checkResult = 0;
    
    training_data_file = 'trainingdata.mat';
    %load(training_data_file, 'trainingData'); 
    %load(training_data_file, 'class_trainingData');
    
    learn_data = trainingData(1:5500,:);
    learn_classes = class_trainingData(1:5500);
    test_data = trainingData(5501:end,:);
    test_classes = class_trainingData(5501:end); %#ok<COLND>
    clear trainingData;
    clear class_trainingData;

    [learnDataAmount, ~] = size(learn_data);
    [learnClassAmount, ~] = size(learn_classes);
    if learnDataAmount == learnClassAmount
        display('learn size maches');
    end
    
    [testDataAmount, ~] = size(test_data);
    [testClassAmount, ~] = size(test_classes);
    if testDataAmount == testClassAmount
        display('test size maches');
    end
    
    mySom = trainClassifier(learn_data, learn_classes);
    
    if(checkResult)
        correct = 0;

        for i = 1:testDataAmount 
            winnerclass = getWinnerClass(mySom, test_data(i,:));
            realWinner = test_classes(i);
            
            if realWinner == winnerclass
                correct = correct + 1;
            end
        end
        correctnessRate = correct / testDataAmount;
        display(['Correctness rate ', num2str(correctnessRate)]);
    else
        correct = 0;
        results = evaluateClassifier(test_data, mySom);

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
    %}
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
function nick = getNickName()
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
    %samples = standardize(samples);
    [sampleAmount, vecLength] = size(samples);
    [classesAmount, ~] = size(classes);
    
    clusters = 64; 
    decay_rate = 0.96; % default 0.96
    min_alpha = 0.01; % default 0.01
    radius_reduction =  0.023; % default 0.023

    mySom = SomClass(clusters, vecLength, min_alpha, decay_rate, radius_reduction);
    mySom = training(mySom, samples);
    mySom = setClasses(mySom, samples, classes);
    
    parameters = mySom;
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
    mySom = parameters;
    
    [testDataAmount, ~] = size(samples);
    results = zeros(testDataAmount, 1);

    for i = 1:testDataAmount 
        winnerclass = getWinnerClass(mySom, samples(i,:));
        results(i) = winnerclass;
    end
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
function d = myDistanceFunction( x, y )
    d = sqrt( sum( (x - y).^2 ) );
end

% From ex7
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

% SOM starts
% based on http://mnemstudio.org/ai/nn/som_python_ex2.txt
function obj = SomClass(clusters, vectorLength, minAlpha, decayRate, reductionPoint)
    if (nargin == 5)
        obj.mClusters = clusters;
        obj.mVectorLength = vectorLength;
        obj.mMinAlpha = minAlpha;
        obj.mDecayRate = decayRate;
        obj.mReductionPoint = reductionPoint;
        obj.mWeightArray = rand(obj.mClusters, obj.mVectorLength);
        obj.mWinnerClasses = zeros(obj.mClusters, 1);
        obj.mDeltaVector = zeros(obj.mClusters, 1);
        obj.mAlpha = 0.6;
    else
        disp('somclass - wrong argument count');
    end
end

function obj = compute_input(obj, vector)
    obj.mDeltaVector = zeros(obj.mClusters, 1);
    for i = 1:obj.mClusters
        d = obj.mWeightArray(i,:) - vector;
        obj.mDeltaVector(i) = sum(d.^2);
    end
end

function minimum = get_minimum(nodeArray)
    [~, minimum] = min(nodeArray); % save index to minimum
end

function obj = setClasses(obj, patternArray, classes)
    winAmountByCluster = cell(obj.mClusters, 1);
    for i = 1:size(patternArray, 1)
        obj = compute_input(obj, patternArray(i,:));
        winner = get_minimum(obj.mDeltaVector);
        winAmountByCluster{winner} = [winAmountByCluster{winner}, classes(i)];
    end
    for i = 1:obj.mClusters
        mostFrequentClass = mode(winAmountByCluster{i},2);
        if(isscalar(mostFrequentClass))
            obj.mWinnerClasses(i) = mostFrequentClass;
        end
    end
end

function winner_class = getWinnerClass(obj, vector)
    obj = compute_input(obj, vector);
    cluster = get_minimum(obj.mDeltaVector);
    winner_class = obj.mWinnerClasses(cluster);
end

function obj = training(obj, patternArray)
    iterations = 0;
    reductionFlag = false;
    reductionPoint = 0;

    while obj.mAlpha > obj.mMinAlpha
        iterations = iterations + 1;
        for i = 1:size(patternArray, 1)
            obj = compute_input(obj, patternArray(i,:));
            dMin = get_minimum(obj.mDeltaVector);
            obj = update_weights(obj, i, dMin, patternArray);
        end
        % Reduce the learning rate.
        obj.mAlpha = obj.mDecayRate * obj.mAlpha;

        % Reduce radius at specified point.
        if obj.mAlpha < obj.mReductionPoint
            if reductionFlag == false
                reductionFlag = true;
                reductionPoint = iterations;
            end
        end
    end
    %{
    display(['Iterations: ', num2str(iterations)  ])
    display(['Neighborhood radius reduced after ', num2str(reductionPoint), ' iterations'])
    display(['obj mAlpha: ', num2str(obj.mAlpha)  ])
    display(['obj mMinAlpha ', num2str(obj.mMinAlpha) ])
    %}
end

function obj = update_weights(obj, vectorNumber, dMin, patternArray)
    for i = 1:obj.mVectorLength
        % Update the winner.
        obj.mWeightArray(dMin, i) = obj.mWeightArray(dMin, i) + (obj.mAlpha * (patternArray(vectorNumber, i) - obj.mWeightArray(dMin, i)));

        % Only include neighbors before radius reduction point is reached.
        if obj.mAlpha > obj.mReductionPoint
            if (dMin > 1) && (dMin < (obj.mClusters)) % TODO : CHECK THIS 
                % Update neighbor to the left...
                obj.mWeightArray(dMin - 1, i) = obj.mWeightArray(dMin - 1, i) + (obj.mAlpha * (patternArray(vectorNumber, i) - obj.mWeightArray(dMin - 1, i)));
                % and update neighbor to the right.
                obj.mWeightArray(dMin + 1, i) = obj.mWeightArray(dMin + 1, i) + (obj.mAlpha * (patternArray(vectorNumber, i) - obj.mWeightArray(dMin + 1, i)));
            else
                if dMin == 1 % TODO : CHECK
                    % Update neighbor to the right.
                    obj.mWeightArray(dMin + 1, i) = obj.mWeightArray(dMin + 1, i) + (obj.mAlpha * (patternArray(vectorNumber, i) - obj.mWeightArray(dMin + 1, i)));
                else
                    % Update neighbor to the left.
                    obj.mWeightArray(dMin - 1, i) = obj.mWeightArray(dMin - 1, i) + (obj.mAlpha * (patternArray(vectorNumber, i) - obj.mWeightArray(dMin - 1, i)));

                end
            end
        end
    end
end           
   