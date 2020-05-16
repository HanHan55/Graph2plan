function [newBox, constraint] = align_adjacent_room3(box, tempBox, updated, type, threshold)
% position of box1 relative to box2
% 0 left-above
% 1 left-below
% 2 left-of
% 3 above
% 4 inside
% 5 surrounding
% 6 below
% 7 right-of
% 8 right-above
% 9 right-below


newBox = box;
constraint = zeros(4, 2);
idx = 1;

if type == 0
    alignV(true);
    alignH(true);
elseif type == 1
    alignV(true);
    alignH(false);
elseif type == 2
    align([2,1], [1,3], threshold);
    align([2,2], [1,2], threshold/2);
    align([2,4], [1,4], threshold/2);
elseif type == 3
    align([2,2], [1,4], threshold);
    align([2,1], [1,1], threshold/2);
    align([2,3], [1,3], threshold/2);
elseif type == 4
    align([2,1], [1,1], true);
    align([2,2], [1,2], true);
    align([2,3], [1,3], true);
    align([2,4], [1,4], true);
elseif type == 5
    align([1,1], [2,1], true);
    align([1,2], [2,2], true);
    align([1,3], [2,3], true);
    align([1,4], [2,4], true);
elseif type == 6
    align([2,4], [1,2], threshold);
    align([2,1], [1,1], threshold/2);
    align([2,3], [1,3], threshold/2);
elseif type == 7
    align([2,3], [1,1], threshold);
    align([2,2], [1,2], threshold/2);
    align([2,4], [1,4], threshold/2);
elseif type == 8
    alignV(false);
    alignH(true);
elseif type == 9
    alignV(false);
    alignH(false);
end

constraint = constraint(1:idx-1, :);

function alignV(isLeft)
    if isLeft
        idx1 = 1;
        idx2 = 3;
    else
        idx1 = 3;
        idx2 = 1;
    end
    
    if abs(tempBox(2,idx1) - tempBox(1,idx2)) <= abs(tempBox(2,idx2) - tempBox(1,idx2))
        align([2,idx1], [1,idx2], threshold/2)
    else
        align([2,idx2], [1,idx2], threshold/2)
    end
end

function alignH(isAbove)
    if isAbove
        idx1 = 2;
        idx2 = 4;
    else
        idx1 = 4;
        idx2 = 2;
    end
    
    if abs(tempBox(2,idx1) - tempBox(1,idx2)) <= abs(tempBox(2,idx2) - tempBox(1,idx2))
        align([2,idx1], [1,idx2], threshold/2)
    else
        align([2,idx2], [1,idx2], threshold/2)
    end
end

function align(idx1, idx2, threshold, attach)
    if nargin < 4
        attach = false;
    end
    if abs(tempBox(idx1(1),idx1(2))- tempBox(idx2(1), idx2(2))) <= threshold    
        if updated(idx1(1), idx1(2)) && ~updated(idx2(1), idx2(2))
            newBox(idx2(1), idx2(2)) = newBox(idx1(1),idx1(2));
        elseif updated(idx2(1), idx2(2)) && ~updated(idx1(1), idx1(2))
            newBox(idx1(1), idx1(2)) = newBox(idx2(1),idx2(2));
        elseif ~updated(idx1(1), idx1(2)) && ~updated(idx2(1), idx2(2))
            if attach
                newBox(idx2(1), idx2(2)) = newBox(idx1(1),idx1(2));
            else
                y = (newBox(idx1(1),idx1(2)) + newBox(idx2(1), idx2(2)))/2;
                newBox(idx1(1),idx1(2)) = y;
                newBox(idx2(1), idx2(2)) = y;
            end
        end
        
        if idx1(1) == 1
            constraint(idx, :) = [idx1(2) idx2(2)];
        else
            constraint(idx, :) = [idx2(2) idx1(2)];
        end
        idx = idx + 1;
    end
end

end