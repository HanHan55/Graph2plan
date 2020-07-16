function [constraint, box, updated] = align_with_boundary(box, boundary, threshold, rType)
tempBox = box;
updated = false(size(box));
closedSeg = zeros(size(box));
distSeg = zeros(size(box));
for i = 1:length(box)
    [closedSeg(i,:), distSeg(i,:)] = find_close_seg(box(i,:), boundary); 
end


box(distSeg <= threshold) = closedSeg(distSeg <= threshold);
updated(distSeg <= threshold) = true;
idx = find(distSeg <= threshold);
constraint = [idx closedSeg(idx)];


% check if any room box blocks the door
entranceBox = get_entrance_space(boundary(1:2, 1:2), boundary(1,3), threshold);
entrancePoly = polyshape(entranceBox([1 1 3 3]), entranceBox([2 4 4 2]));
for i = 1:length(box)
    if rType(i) ~= 10 && rType(i) ~= 0
        roomPoly = polyshape(box(i, [1 1 3 3]), box(i, [2 4 4 2]));
        if overlaps(entrancePoly, roomPoly)
            box(i,:) = shrink_box(roomPoly, entrancePoly, boundary(1,3));
            updated(i, box(i,:)==tempBox(i,:)) = false;
            updated(i, box(i,:)~=tempBox(i,:)) = true;
        end
    end        
end
