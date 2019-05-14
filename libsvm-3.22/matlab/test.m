clc
clear all
close all

feature1 = textread('feature1.txt','%d','delimiter', ',');
feature1 = reshape(feature1,[140 1000]);
feature1 = feature1';
feature1 = feature1(:,2:end);
% 99

feature2 = textread('feature2.txt','%d','delimiter', ',');
feature2 = reshape(feature2,[73 1000]);
feature2 = feature2';
feature2 = feature2(:,2:end);
% % 99.5
% 
feature3 = textread('feature3.txt','%d','delimiter', ',');
feature3 = reshape(feature3,[279 1000]);
feature3 = feature3';
feature3 = feature3(:,2:end);
% % 98.5
% 
feature4 = textread('feature4.txt','%d','delimiter', ',');
feature4 = reshape(feature4,[140 1000]);
feature4 = feature4';
feature4 = feature4(:,2:end);
% % 90.5
% 
feature5 = textread('feature5.txt','%d','delimiter', ',');
feature5 = reshape(feature5,[12*23+1 1000]);
feature5 = feature5';
feature5 = feature5(:,2:end);
% % 99.5
% 
feature6 = textread('feature6.txt','%d','delimiter', ',');
feature6 = reshape(feature6,[8*16+1 1000]);
feature6 = feature6';
feature6 = feature6(:,2:end);
% % 100
% 
% feature = [3*feature1 4*feature2 2*feature3 feature4 4*feature5 100*feature6];
feature = feature1;
% feature = [2*feature1 4*feature2 2.2*feature3];
% feature = feature6;
% w=[[] [] [] [] [] []];
classification_num = 13;%26
allclass = [10 11 12 20 22 25 26 28 30 31 32 33 34]% 110 111 112 120 122 125 126 128 130 131 132 133 134];
indexInfo = ['京' '渝' '鄂' '0'  '2' '5' '6' '8' 'A' 'B' 'C' 'D' 'Q']% '京' '渝' '鄂' '0'  '2' '5' '6' '8' 'A' 'B' 'C' 'D' 'Q'];
[~, class, name] = textread('Char_Index.txt','%d %d %s',1000, 'headerlines',1);
% transform_index = zeros(1000, 1);%存储1-13类别编号
% for i = 1 : classification_num
%     transform_index( :, 1) = transform_index(:, 1) + (index == class(i)) * i; %将原来的存储编号映射到1-13上
% end
train_num = 800;
% selection_index = (randperm (1000) <= train_num);
% save selection_index.mat selection_index;
load  selection_index.mat;

model = svmtrain( class(selection_index,:),feature(selection_index,:),'-t 1 -d 1 -g 0.1 -r 0');
[predict_label, accuracy, dec_values] = svmpredict(class(~selection_index,:),feature(~selection_index,:), model);

A = class(~selection_index,:);
B = predict_label;
err_index = A~=B;
C = A(err_index);
D = B(err_index);
err_name = name(~selection_index,:);
N = err_name(err_index);
for i = 1:length(C)
    E = find(allclass == C(i));
    F = find(allclass == D(i));
    [indexInfo(E) ' 被误识别成 ' indexInfo(F)]
    figure(i);
    imshow(['E:\Feature Extraction\Char_Image\' N{i}]);
    N{i}
end
%20090105213655843ch3IMG.JPG



