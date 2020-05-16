function order = find_room_order(M)

n = size(M,1);
G = digraph(M);
name = cell(n,1);
for i = 1:n
    name{i} = num2str(i);
end
G.Nodes.Name = name;

order = zeros(n, 1);
i = 1;
while i <= n
    D = indegree(G);
    c = find(D==0);
    if isempty(c)
        idx = find(D==1);
        c = setdiff(idx, order);
        order(i) = str2double(G.Nodes.Name{c(1)});
        G = rmnode(G, c(1));
        i = i+1;
    else
        for j = 1:length(c)
            order(i+j-1) = str2double(G.Nodes.Name{c(j)});
        end
        G = rmnode(G, c);
        i = i + length(c);
    end
end