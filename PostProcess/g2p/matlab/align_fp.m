function [newBox, order, rBoundary] = align_fp(boundary, rBox, rType, rEdge, fp, threshold, drawResult)
% align the neighboring rooms first and then align with the boundary

if nargin < 7
    drawResult =false;
end

% pre-processing: 
% move the edge relation w.r.t. living room to the end
livingIdx = find(rType==0);
idx = rEdge(:,1) == livingIdx-1 | rEdge(:,2) == livingIdx-1;
% a = rEdge(~idx, :);
% b = rEdge(idx, :);
% rEdge = [a; b];
rEdge = rEdge(~idx, :);
entranceBox = get_entrance_space(boundary(1:2, 1:2), boundary(1,3), threshold);

if drawResult
    clf
    subplot(2,2,1)
    plot_fp(rBox, boundary, rType, entranceBox);
    title('original');
end

%% option #1: use greedy method: align with boundary first and then neighbor
% 1. align with boundary after the neighbors have been aligned
[~, newBox, updated] = align_with_boundary(rBox, boundary, threshold, rType);

if drawResult
    subplot(2,2,2)
    plot_fp(newBox, boundary, rType, entranceBox);
    title('Align with boundary');
end


% 2. for each adjacent pair of room,
[~, newBox, ~] = align_neighbor(newBox, rEdge, updated, threshold+6);
if drawResult
    subplot(2,2,3)
    plot_fp(newBox, boundary, rType, entranceBox);
    title('Align with neighbors');
end

% 3. regularize fp, include crop using boundary, gap filling
[newBox, order] = regularize_fp(newBox, boundary, rType);

% 4. generate the room polygons
[newBox, rBoundary] = get_room_boundary(newBox, boundary, order);
    
if drawResult
    subplot(2,2,4)
    plot_fp(newBox(order,:), boundary, rType(order), entranceBox);
    title('Regularize fp');
end

% %% option #2: use optimization to align neighbors, and then align the boundary 
% % 1. get the constraint from the adjacent rooms， and optimize
% %[constraint1, ~, ~] = align_with_boundary(rBox, boundary, threshold, rNode);
% [constraint2, ~, ~] = align_neighbor(rBox, rEdge, [], threshold+2);
% newBox = optimize_fp(rBox, [], constraint2);
% if drawResult
%     subplot(3,4,6)
%     plot_fp(newBox, boundary, rNode, entranceBox);
%     title('Optimize the neighboring');
% end
% 
% % 2. align with boundary after the neighbors have been aligned
% [constraint1, newBox2, ~] = align_with_boundary(newBox, boundary, threshold, rNode);
% if drawResult
%     subplot(3,4,7)
%     plot_fp(newBox2, boundary, rNode, entranceBox);
%     title('Align with boundary w/o optimization');
% end
% 
% % 3. regularize fp, include crop using boundary, gap filling
% [newBox2, order] = regularize_fp(newBox2, boundary, rNode);
% if drawResult
%     subplot(3,4,8)
%     plot_fp(newBox2(order,:), boundary, rNode(order,:), entranceBox);
%     title('Regularize fp');
% end
% 
% 
% 
% newBox = optimize_fp(newBox, constraint1, constraint2);
% if drawResult
%     subplot(3,4,11)
%     plot_fp(newBox, boundary, rNode, entranceBox);
%     title('Align with boundary with optimization');
% end
% 
% % 3. regularize fp, include crop using boundary, gap filling
% [newBox, order] = regularize_fp(newBox, boundary, rNode);
% if drawResult
%     subplot(3,4,12)
%     plot_fp(newBox(order,:), boundary, rNode(order,:), entranceBox);
%     title('Regularize fp');
%     if ~isempty(figName)
%         saveas(gcf, figName);
%     end
% end
% 



%%
end

