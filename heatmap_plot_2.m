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
vcns_lin = [0:2:14];
vcns_size = size(vcns_lin,2)-1;

vcns_1d = zeros(1,vcns_size);
vcns_1d_make = zeros(1,vcns_size);
for idx_entries = 1:17805 
    for i = 1:vcns_size
        if v_cns{idx_entries,1} >= vcns_lin(i) && v_cns{idx_entries,1} < vcns_lin(i+1)
            vcns_1d(i) = vcns_1d(i) + 1;
        end
        
        if v_cns{idx_entries,1} >= vcns_lin(i) && v_cns{idx_entries,1} < vcns_lin(i+1) && points{idx_entries,1} == 3
            vcns_1d_make(i) = vcns_1d_make(i) + 1;
        end 
    end
end

%% first plot
bar(vcns_1d)
xticklabels({'0','2','4','6','8','10','12'})
xlabel('velocity of c&s (x ft/s to x+2 ft/s)')
ylabel('counts')

%% second plot
bar(vcns_1d_make)
xticklabels({'0','2','4','6','8','10','12'})
xlabel('velocity of c&s (x ft/s to x+2 ft/s)')
ylabel('counts')


%% third plot
bar(vcns_1d_make./vcns_1d)
xticklabels({'0','2','4','6','8','10','12'})
xlabel('velocity of c&s (x ft/s to x+2 ft/s)')
ylabel('fraction made')
ylim([0.25 0.4])

%% 
tcns_lin = [0:1:10];
tcns_size = size(tcns_lin,2)-1;

tcns_1d = zeros(1,tcns_size);
tcns_1d_make = zeros(1,tcns_size);
for idx_entries = 1:17805 
    for i = 1:tcns_size
        if t_cns{idx_entries,1} >= tcns_lin(i) && t_cns{idx_entries,1} < tcns_lin(i+1)
            tcns_1d(i) = tcns_1d(i) + 1;
        end
        
        if t_cns{idx_entries,1} >= tcns_lin(i) && t_cns{idx_entries,1} < tcns_lin(i+1) && points{idx_entries,1} == 3
            tcns_1d_make(i) = tcns_1d_make(i) + 1;
        end 
    end
end

%% 4th plot
bar(tcns_1d)
xticks([1:1:10])
xticklabels({'0','1','2','3','4','5','6','7','8','9'})
xlabel('time of c&s (x sec to x+1 sec)')
ylabel('counts')

%% fifth plot
bar(tcns_1d_make)
xticks([1:1:10])
xticklabels({'0','1','2','3','4','5','6','7','8','9'})
xlabel('time of c&s (x sec to x+1 sec)')
ylabel('counts')

%% sixth plot
bar(tcns_1d_make./tcns_1d)
xticks([1:1:10])
xticklabels({'0','1','2','3','4','5','6','7','8','9'})
xlabel('time of c&s (x sec to x+1 sec)')
ylabel('fraction made')
ylim([0.25 0.45])


%% 2d plot 
vcns_lin = [0:2:14];
vcns_size = size(vcns_lin,2)-1;

tcns_lin = [0:1:10];
tcns_size = size(tcns_lin,2)-1;

% j is y is tcns
% i is x is vcns
mat_tcns_vs_vcns = zeros(tcns_size, vcns_size)
mat_tcns_vs_vcns_make = zeros(tcns_size, vcns_size)

for idx_entries = 1:17805 
    for i = 1:vcns_size
        for j = 1:tcns_size
            if v_cns{idx_entries,1} >= vcns_lin(i) && v_cns{idx_entries,1} < vcns_lin(i+1) && t_cns{idx_entries,1} >= tcns_lin(j) && t_cns{idx_entries,1} < tcns_lin(j+1)
                mat_tcns_vs_vcns(j,i) = mat_tcns_vs_vcns(j,i) + 1;
            end

            if v_cns{idx_entries,1} >= vcns_lin(i) && v_cns{idx_entries,1} < vcns_lin(i+1) && t_cns{idx_entries,1} >= tcns_lin(j) && t_cns{idx_entries,1} < tcns_lin(j+1) && points{idx_entries,1} == 3
                mat_tcns_vs_vcns_make(j,i) = mat_tcns_vs_vcns_make(j,i) + 1;
            end 
        end
    end
end
%%
heatmap(mat_tcns_vs_vcns)
ax = gca;
ax.XData = {'0','2','4','6','8','10','12'}
ax.YData = {'0','1','2','3','4','5','6','7','8','9'}
xlabel('velocity of c&s (x ft/s to x+2 ft/s)')
ylabel('time of c&s (x sec to x+1 sec)')

%%
a = mat_tcns_vs_vcns_make./mat_tcns_vs_vcns
a(mat_tcns_vs_vcns <= 20) = NaN
heatmap(a)
ax = gca;
ax.Title = "showing results if there are over 20 shots"
ax.XData = {'0','2','4','6','8','10','12'}
ax.YData = {'0','1','2','3','4','5','6','7','8','9'}
xlabel('velocity of c&s (x ft/s to x+2 ft/s)')
ylabel('time of c&s (x sec to x+1 sec)')



