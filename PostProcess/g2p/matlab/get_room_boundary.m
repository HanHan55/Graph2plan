function [newBox, rBoundary] = get_room_boundary(box, boundary, order)

isNew = boundary(:,4);
polyBoundary = polyshape(boundary(~isNew,1),  boundary(~isNew,2));

poly = cell(size(box,1), 1);
for i = 1:size(box,1)
    poly{i} = polyshape(box(i, [1 1 3 3]), box(i, [2 4 4 2]));
end

newBox = box;
rBoundary = cell(size(box,1), 1);
for i = 1:size(box,1)
    idx = order(i);
    
    rPoly = intersect(polyBoundary, poly{idx}); 
    for j = i+1:size(box,1)
        rPoly = subtract(rPoly, poly{order(j)});
    end
    rBoundary{idx} = rPoly.Vertices;
    [xLimit, yLimit]= boundingbox(rPoly);
    if ~isempty(xLimit)
        newBox(idx,:) = [xLimit(1) yLimit(1) xLimit(2), yLimit(2)];
    end
end