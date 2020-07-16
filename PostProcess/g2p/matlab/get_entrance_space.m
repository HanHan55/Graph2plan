
function doorBox = get_entrance_space(doorSeg, doorOri, threshold)

doorBox = [doorSeg(1,:) doorSeg(2,:)];
if doorOri == 0
    doorBox(4) = doorBox(4) + threshold;
elseif doorOri == 1
    doorBox(1) = doorBox(1) - threshold;
elseif doorOri == 2
    doorBox(2) = doorBox(2) - threshold;
elseif doorOri == 3
    doorBox(3) = doorBox(3) + threshold;
end
    