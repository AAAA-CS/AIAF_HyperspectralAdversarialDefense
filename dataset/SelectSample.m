
%==============================================
clc
clear all

inputlabel = ['E:\matlab_pytorch\PaviaU01\PaviaU_gt.mat'];
load(inputlabel);
Img_GT = paviaU_gt;
[m,n] = size(Img_GT);
imggt = reshape(Img_GT, [1, m*n]);
%======select training sample=============
% pathTrain = 'E:\MD learning\sample\image4\TR_5\';
class = max(max(Img_GT));%读出最大的样本类别
TRLabel = zeros(size(Img_GT,1),size(Img_GT,2));
TSLabel = zeros(size(Img_GT,1),size(Img_GT,2));
for c = 1:class
    TestNO = 1000;
    TrainNO = 900;
    num = TrainNO + TestNO;
    [Index,indey] = find(Img_GT == c);
    k1 = randperm(length(Index));
    PoX1 = Index(k1(1:TrainNO));
    PoY1 = indey(k1(1:TrainNO));
    [y,shenyu] = size(k1(TrainNO+1:end));

    TestNO = shenyu;
    num = TrainNO + shenyu;
    PoX2 = Index(k1(TrainNO+1:num));
    PoY2 = indey(k1(TrainNO+1:num));
    for i = 1:length(PoX1)
        TRLabel(PoX1(i),PoY1(i)) = Img_GT(PoX1(i),PoY1(i));
    end
    %================================
    for i = 1:length(PoX2)
        TSLabel(PoX2(i),PoY2(i)) = Img_GT(PoX2(i),PoY2(i));
    end

end
save 'TRLabel.mat' TRLabel
save 'TSLabel.mat' TSLabel