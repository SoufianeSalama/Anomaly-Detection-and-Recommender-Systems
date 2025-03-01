function [bestEpsilon bestF1 f1scoreMatrix epsilonMatrix] = selectThreshold(yval, pval)
%SELECTTHRESHOLD Find the best threshold (epsilon) to use for selecting
%outliers
%   [bestEpsilon bestF1] = SELECTTHRESHOLD(yval, pval) finds the best
%   threshold to use for selecting outliers based on the results from a
%   validation set (pval) and the ground truth (yval).
%
f1scoreMatrix = [];
epsilonMatrix = [];

bestEpsilon = 0;
bestF1 = 0;
F1 = 0;
count = 0;
stepsize = (max(pval) - min(pval)) / 1000;
stepsize
for epsilon = min(pval):stepsize:max(pval)
    
    % ====================== YOUR CODE HERE ======================
    % Instructions: Compute the F1 score of choosing epsilon as the
    %               threshold and place the value in F1. The code at the
    %               end of the loop will compare the F1 score for this
    %               choice of epsilon and set it to be the best epsilon if
    %               it is better than the current choice of epsilon.
    %               
    % Note: You can use predictions = (pval < epsilon) to get a binary vector
    %       of 0's and 1's of the outlier predictions

    predictions = (pval < epsilon);
    truepos  = sum((predictions==1) & (yval==1));
    falsepos = sum((predictions==1) & (yval==0));
    falseneg = sum((predictions==0) & (yval==1));

    precision = truepos/(truepos+falsepos);
    recall = truepos/(truepos+falseneg);
    F1 = (2*precision*recall)/(precision+recall);
    
    f1scoreMatrix(:,end+1) = F1;
    epsilonMatrix(:,end+1) = epsilon;
    % =============================================================

    if F1 > bestF1
       bestF1 = F1;
       bestEpsilon = epsilon;
    end
    count = count + 1 ;
end
count

% figure; hold on;
% plot(epsilonMatrix, f1scoreMatrix);
% title('1.3: F1 Score in func of epsilon');
% xlabel('Epsilon');
% ylabel('F1 Score');
% hold off;

end
