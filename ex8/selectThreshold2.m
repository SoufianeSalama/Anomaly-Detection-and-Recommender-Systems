function [bestEpsilon bestF1 f1scoreMatrix epsilonMatrix] = selectThreshold2(yval, pval)
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
[pval, index] = sort(pval);
yval=yval(index,:);

for value = 1:1:length(pval)-1
    fprintf('Number at position %d is %d\n', value, pval(value));
    gem = (pval(value) + pval(value+1))/2;
    epsilon = gem; 
    
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
end


end
