function [inhale_volumes,exhale_volumes] = find_respiratory_volumes(resp, srate, inhale_onsets, exhale_onsets)

% estimates the volume of air displaced in all inhales and exhales
% requires zero crosses
% assumes peaks always come first
% assumes respiration has been baseline corrected;

inhale_volumes = zeros(1,length(inhale_onsets));
exhale_volumes = zeros(1,length(exhale_onsets));

for bi = 1:length(exhale_onsets)
    inhale_integral = sum(abs(resp(inhale_onsets(bi):exhale_onsets(bi)))); %11/6 added abs
    inhale_volumes(1,bi)=inhale_integral;
end

for bi = 1:length(exhale_onsets)-1
    exhale_integral = sum(abs(resp(exhale_onsets(bi):inhale_onsets(bi+1)))); %11/6 added abs
    exhale_volumes(1,bi)=exhale_integral;
end

% last exhale is different because there is no following inhale
exhale_window = resp(exhale_onsets(end):end);
pre_putative_exhale_end = find(exhale_window>0,1,'first');

% check that there is a real exhale end
if isempty(pre_putative_exhale_end) || pre_putative_exhale_end<srate/2;
    % volume cannot be calculated
    exhale_volumes(end)=nan;
else
    putative_exhale_end = pre_putative_exhale_end+exhale_onsets(end);
    exhale_integral = sum(abs(resp(exhale_onsets(end):putative_exhale_end)));
    high_check = exhale_integral < mean(exhale_volumes(1:end-1))+2*std(exhale_volumes(1:end-1));
    low_check = exhale_integral > mean(exhale_volumes(1:end-1))-2*std(exhale_volumes(1:end-1));
    if low_check && high_check
        exhale_volumes(end)=exhale_integral;
    else
        exhale_volumes(end)=nan;
    end
end

% normalize for different sampling rates
inhale_volumes = (inhale_volumes/srate)*1000;
exhale_volumes = (exhale_volumes/srate)*1000;
