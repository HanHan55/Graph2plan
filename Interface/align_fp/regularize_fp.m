function [box, order] = regularize_fp(box, boundary, rType)

% 1. use the boundary to crop each room box
isNew = boundary(:,4);
polyBoundary = polyshape(boundary(~isNew,1),  boundary(~isNew,2));
for i = 1:size(box, 1)
    polyRoom = polyshape(box(i, [1 1 3 3]), box(i, [2 4 4 2]));
    [xLimit, yLimit] = boundingbox(intersect(polyBoundary,polyRoom));
    if isempty(xLimit)
        disp('One room outside the building!'); 
    else
        box(i,:) = [xLimit(1) yLimit(1) xLimit(2), yLimit(2)];
    end
end


% 2. check if there is any overlapped region to determine the layer of boxes
orderM = false(size(box,1), size(box,1));
for i = 1:size(box,1)
    polyRoom1 = polyshape(box(i, [1 1 3 3]), box(i, [2 4 4 2]));
    area1 = area(polyRoom1);
    for j = i+1:size(box,1)
         polyRoom2 = polyshape(box(j, [1 1 3 3]), box(j, [2 4 4 2]));
         area2 = area(polyRoom2);
         inter = intersect(polyRoom1, polyRoom2);
         if inter.NumRegions >= 1
             if area1 <= area2 % may need to add the FP into consideration
                 orderM(i,j) = true;
             else
                 orderM(j,i) = true;
             end
         end
    end
end
order = 1:size(box,1);
if any(orderM(:))
    order = find_room_order(orderM);
end
order = order(end:-1:1);

% 3. check if there are more than one uncovered regions inside the building
livingIdx = find(rType==0);
for i = 1:size(box, 1)
    if i ~= livingIdx
       if box(i,1)==box(i,3) || box(i,2)==box(i,4)
           disp('Empty box!!!');
       else
           polyRoom = polyshape(box(i, [1 1 3 3]), box(i, [2 4 4 2]));
           polyBoundary = subtract(polyBoundary,polyRoom);
       end
       
    end
end
livingPoly = polyshape(box(livingIdx, [1 1 3 3]), box(livingIdx, [2 4 4 2]));

gap = polyBoundary;
if gap.NumRegions == 1
    [xLimit, yLimit] = boundingbox(gap);
    box(livingIdx,:) = [xLimit(1) yLimit(1) xLimit(2), yLimit(2)];
else
    rIdx = find(isnan(gap.Vertices(:,1)));
    rIdx = [rIdx; size(gap.Vertices,1)+1];
    
    % for each region, check if it intersects with the living room, 
    % otherwise get the room label and find the room that should cover 
    % the region
    
    region = cell(length(rIdx), 1);
    overlapArea = zeros(length(rIdx), 1);
    closeRoomIdx = zeros(length(rIdx), 1);
    idx = 1;
    for k = 1:length(rIdx)
        regionV = gap.Vertices(idx:rIdx(k)-1, :);
        idx = rIdx(k) + 1;
        region{k} = polyshape(regionV);
        
        if overlaps(region{k}, livingPoly)
            iter = intersect(region{k}, livingPoly);
            overlapArea(k) = area(iter);
        end
        
        [x, y] = centroid(region{k});
        center = [x, y];
        
        dist = 256;
        bIdx = 0;
        for i = 1:size(box, 1)
            b = box(i, :);
            bCenter = double([(b(:,1)+b(:,3))/2, (b(:,2)+b(:,4))/2]);
            d = norm(bCenter-center);
            if d<dist
                dist = d;
                bIdx = i;
            end
        end
        closeRoomIdx(k) = bIdx;
    end 
    
    [~, lIdx] = max(overlapArea);
    for k = 1:length(closeRoomIdx)
        if k == lIdx
            [xLimit, yLimit] = boundingbox(region{k});
            box(livingIdx,:) = [xLimit(1) yLimit(1) xLimit(2), yLimit(2)];
        else
            room = polyshape(box(closeRoomIdx(k), [1 1 3 3]), box(closeRoomIdx(k), [2 4 4 2]));
            [xLimit, yLimit] = boundingbox(union(room, region{k}));
            box(closeRoomIdx(k),:) = [xLimit(1) yLimit(1) xLimit(2), yLimit(2)];
        end
    end        
end
    
