data = importdata("threes.mat");
points = data(:,11);
distance = data(:,14);
defender_d = data(:,19);

distance_lin = [21:1:30];
defender_d_lin = [0:0.5:10];

d_size = size(distance_lin,2)-1;
d_d_size = size(defender_d_lin,2)-1;

matrix_d_vs_dd = zeros(d_d_size, d_size);

d_1d = zeros(1,d_size);
for idx_entries = 1:17805 
    for i = 1:d_size
        if distance{idx_entries,1} >= distance_lin(i) && distance{idx_entries,1} < distance_lin(i+1)
            d_1d(i) = d_1d(i) + 1;
        end
    end
end