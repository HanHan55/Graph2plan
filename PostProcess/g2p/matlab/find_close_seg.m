function [closedSeg, distSeg, idx] = find_close_seg(box, boundary)

% need to carefully select the closed wall seg for each box
% cannot introduce a hole inside the boundary

isNew = boundary(:,4);
boundary = double(boundary(~isNew, :));

% get the ordered horizontal and vertical segments on the boundary
bSeg = [boundary(:, 1:2), boundary([2:end 1], 1:2), boundary(:,3)];
vSeg = bSeg(mod(boundary(:,3), 2)==1, :);
vSeg(vSeg(:,5)==3, [2 4]) = vSeg(vSeg(:,5)==3, [4 2]);
[~, I] = sort(vSeg(:,1));
vSeg = vSeg(I,:);

hSeg = bSeg(mod(boundary(:,3), 2)==0, :);
hSeg(hSeg(:,5)==2, [1 3]) = hSeg(hSeg(:,5)==2, [3 1]);
[~, I] = sort(hSeg(:,2));
hSeg = hSeg(I,:);

closedSeg = ones(1,4)*256;
distSeg = ones(1,4)*256;
idx = zeros(1, 4);

% check vertial seg
for i = 1:size(vSeg,1)
    seg = vSeg(i, :);
    vdist = 0;
    if seg(4) <= box(2) 
        vdist = box(2) - seg(4);
    elseif seg(2) >= box(4) 
        vdist = seg(2) - box(4);
    end
    
    hdist = box([1 3]) - seg(1);
    dist1 = norm(double([hdist(1), vdist])); 
    dist3 = norm(double([hdist(2), vdist])); 
    
    if dist1 < distSeg(1) && dist1 <= dist3 &&  hdist(1) > 0 
        distSeg(1) = dist1;
        idx(1) = i;
        closedSeg(1) = seg(1);
    elseif dist3 < distSeg(3) && hdist(2) < 0 
        distSeg(3) = dist3;
        idx(3) = i;
        closedSeg(3) = seg(3);
    end
end

% check horizontal seg
for i = 1:size(hSeg,1)
    
    seg = hSeg(i, :);
    hdist = 0;
    if seg(3) <= box(1) 
        hdist = box(1) - seg(3);
    elseif seg(1) >= box(3) 
        hdist = seg(1) - box(3);
    end
    
    vdist = box([2 4]) - seg(2);
    dist2 = norm(double([vdist(1), hdist]));
    dist4 = norm(double([vdist(2), hdist]));
    
    if dist2 <= dist4 && dist2 < distSeg(2) &&  vdist(1) > 0 
        distSeg(2) = dist2;
        idx(2) = i;
        closedSeg(2) = seg(2);
    elseif dist4 < distSeg(4) && vdist(2) < 0 
        distSeg(4) = dist4;
        idx(4) = i;
        closedSeg(4) = seg(4);
    end
end