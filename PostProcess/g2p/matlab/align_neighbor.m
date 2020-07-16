function [constraint, box, updated] = align_neighbor(box, rEdge, updated, threshold)

if isempty(updated)
    updated = false(size(box));
end

tempBox = box;
constraint = zeros(size(rEdge, 1)*3, 2); 
iBegin = 1;
checked = false(size(rEdge, 1), 1);
updatedCount = get_updated_count(updated, rEdge);
for i = 1:size(rEdge, 1)
    I = find(~checked);
    [~, t] = maxk(updatedCount(I), 1);
    checked(I(t)) = true;
    idx = rEdge(I(t),1:2)+1;
    [b, c] = align_adjacent_room3(box(idx, :), tempBox(idx, :), updated(idx,:), rEdge(I(t),3), threshold);
    for j = 1:length(idx)
        
        updated(idx(j), c(:,j)) = true;
        
        c(:, j) = (c(:,j)-1)*size(box,1) + double(idx(j)); 
        
        if b(j, 1) == b(j, 3)
            b(j, [1 3]) = box(idx(j), [1 3]);
            updated(idx(j), c(:,j)) = false;
        end
        if b(j, 2) == b(j, 4)
            b(j, [2 4]) = box(idx(j), [2 4]);
            updated(idx(j), c(:,j)) = false;
        end
        
    end
    box(idx, :) = b;
    
    
    cNum = size(c, 1);
    
    constraint(iBegin:iBegin+cNum-1, :) = c;
    iBegin = iBegin+cNum;
    
    updatedCount = get_updated_count(updated, rEdge);
end
constraint = constraint(1:iBegin-1, :);

function updatedCount = get_updated_count(updated, rEdge)
    updatedCount = zeros(size(rEdge, 1), 1);
    for k = 1:size(rEdge, 1)
        index = rEdge(k,1:2)+1;
        updatedCount(k) = sum(sum(updated(index,:)));
    end
end
end