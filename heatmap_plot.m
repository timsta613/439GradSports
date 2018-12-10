data = importdata("threes.mat");
points = data(:,11);
distance = data(:,14);
defender_d = data(:,19);
home = data(:,5)
shot_angle = data(:,15)
v_cns = data(:,13)
defender_v = data(:,20)
t_cns = data(:,12)
shooter_movement = data(:,17)

%% 
distance_lin = [21:1:30];
d_size = size(distance_lin,2)-1;

d_1d = zeros(1,d_size);
d_1d_make = zeros(1,d_size);
for idx_entries = 1:17805 
    for i = 1:d_size
        if distance{idx_entries,1} >= distance_lin(i) && distance{idx_entries,1} < distance_lin(i+1)
            d_1d(i) = d_1d(i) + 1;
        end
        
        if distance{idx_entries,1} >= distance_lin(i) && distance{idx_entries,1} < distance_lin(i+1) && points{idx_entries,1} == 3
            d_1d_make(i) = d_1d_make(i) + 1;
        end 
    end
end

%% first plot
bar(d_1d)
xticklabels({'21','22','23','24','25','26','27','28','29'})
xlabel('distance of shots (x feet to x+1 feet)')
ylabel('counts')

%% second plot
bar(d_1d_make)
xticklabels({'21','22','23','24','25','26','27','28','29'})
xlabel('distance of shots (x feet to x+1 feet)')
ylabel('counts')

%% third plot
bar(d_1d_make./d_1d)
xticklabels({'21','22','23','24','25','26','27','28','29'})
xlabel('distance of shots (x feet to x+1 feet)')
ylabel('fraction made')
ylim([0.25 0.4])

%% 


d_d_lin = [0:1:20];
d_d_size = size(defender_d_lin,2)-1;

matrix_d_vs_dd = zeros(d_d_size, d_size);

dd_1d = zeros(1,d_d_size);
dd_1d_make = zeros(1,d_d_size);

for idx_entries = 1:17805 
    for i = 1:d_d_size
        if defender_d{idx_entries,1} >= d_d_lin(i) && defender_d{idx_entries,1} < d_d_lin (i+1)
            dd_1d(i) = dd_1d(i) + 1;
        end
        
        if defender_d{idx_entries,1} >= d_d_lin (i) && defender_d{idx_entries,1} < d_d_lin (i+1) && points{idx_entries,1} == 3
            dd_1d_make(i) = dd_1d_make(i) + 1;
        end 
    end
end

%% 4th plot
bar(dd_1d)
xticks([1:1:20])
xticklabels({'0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19'})
xlabel('distance of defender (x feet to x+1 feet)')
ylabel('counts')

%% fifth plot
bar(dd_1d_make)
xticks([1:1:20])
xticklabels({'0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19'})
xlabel('distance of defender (x feet to x+1 feet)')
ylabel('counts')

%% sixth plot
bar(dd_1d_make./dd_1d)
xticks([1:1:20])
xticklabels({'0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19'})
xlabel('distance of defender (x feet to x+1 feet)')
ylabel('fraction made')
ylim([0.25 0.45])


%% 2d plot 
d_d_lin = [0:2:18];
d_d_size = size(d_d_lin,2)-1;


distance_lin = [22:2:30];
d_size = size(distance_lin,2)-1;

% j is y is defender distance
% i is x is player distance
mat_dd_vs_d = zeros(d_d_size, d_size)
mat_dd_vs_d_make = zeros(d_d_size, d_size)

for idx_entries = 1:17805 
    for i = 1:d_size
        for j = 1:d_d_size
            if distance{idx_entries,1} >= distance_lin(i) && distance{idx_entries,1} < distance_lin (i+1) && defender_d{idx_entries,1} >= d_d_lin(j) && defender_d{idx_entries,1} < d_d_lin(j+1)
                mat_dd_vs_d(j,i) = mat_dd_vs_d(j,i) + 1;
            end

            if distance{idx_entries,1} >= distance_lin(i) && distance{idx_entries,1} < distance_lin (i+1) && defender_d{idx_entries,1} >= d_d_lin(j) && defender_d{idx_entries,1} < d_d_lin(j+1) && points{idx_entries,1} == 3
                mat_dd_vs_d_make(j,i) = mat_dd_vs_d_make(j,i) + 1;
            end 
        end
    end
end
%%
heatmap(mat_dd_vs_d)
ax = gca;
ax.XData = {'22','24','26','28'}
ax.YData = {'0','2','4','6','8','10','12','14','16'}
xlabel('distance of shot (x feet to x+2 feet)')
ylabel('distance of defender (x feet to x+2 feet)')

%%
heatmap(mat_dd_vs_d_make./mat_dd_vs_d)
ax = gca;
ax.XData = {'22','24','26','28'}
ax.YData = {'0','2','4','6','8','10','12','14','16'}
xlabel('distance of shot (x feet to x+2 feet)')
ylabel('distance of defender (x feet to x+2 feet)')



