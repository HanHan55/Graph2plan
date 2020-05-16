function box = shrink_box(roomPoly, entrancePoly, doorOrient)

[PG, shapeId, ~] = subtract(roomPoly, entrancePoly);
idx1 = find(shapeId==1);
d = idx1(2:end) - idx1(1:end-1);
i = find(d~=1);
if ~isempty(i)
    idx1 = idx1([i+1:end 1:i]);
end

idx2 = find(shapeId~=1);
d = idx2(2:end) - idx2(1:end-1);
i = find(d~=1);
if ~isempty(i)
    idx2 = idx2([i+1:end 1:i]);
end

remainPoint = length(idx1);
if remainPoint == 2
    box = [min(PG.Vertices) max(PG.Vertices)];
elseif remainPoint == 3
    assert(length(idx2) == 3);
    pointSet1 = PG.Vertices([idx1(1:2); idx2(2)], :);
    pointSet2 = PG.Vertices([idx1(2:3); idx2(2)], :);
    if mod(doorOrient, 2) == 0 % door grow vertically
        if pointSet1(1,1) ==  pointSet1(2,1)
            box = [min(pointSet1) max(pointSet1)];
        else
            box = [min(pointSet2) max(pointSet2)];
        end
    else
        if pointSet1(1,2) ==  pointSet1(2,2)
            box = [min(pointSet1) max(pointSet1)];
        else
            box = [min(pointSet2) max(pointSet2)];
        end
    end
elseif remainPoint == 4 
    % elseif remainPoint == 4 && length(idx2) == 4
%     pointSet = PG.Vertices([idx1(2:3); idx2(2:3)], :);
%     box = [min(pointSet) max(pointSet)];
% elseif remainPoint == 4 % door inside the box
    [x1, y1] = centroid(roomPoly);
    [x2, y2] = centroid(entrancePoly);
    box = [min(roomPoly.Vertices)  max(roomPoly.Vertices)];
    if mod(doorOrient, 2) == 0 % door grow vertically
        if x1 < x2
            box(3) = min(entrancePoly.Vertices(:,1));
        else
            box(1) = max(entrancePoly.Vertices(:,1));
        end
    else
        if y1 < y2
            box(4) = min(entrancePoly.Vertices(:,2));
        else
            box(2) = max(entrancePoly.Vertices(:,2));
        end
    end
else
    disp(['There are other cases with point number = ', num2str(length(shapeId))]);
end