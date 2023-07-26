fid = fopen('N1.txt');
data = fscanf(fid, '%f %f', [2 Inf]);
fclose(fid);
data = data'; 

current = data(:,1);
voltage = data(:,2);


subplot(2,1,1)
plot(current, 'Color', [0 0 1], 'Linewidth', 1);
ylabel('pA')

subplot(2,1,2)
plot(voltage, 'Color', [0 0 1], 'Linewidth', 1);
ylabel('mV')