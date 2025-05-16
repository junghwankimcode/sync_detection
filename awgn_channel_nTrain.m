clear all;
close all;
clc;

%trainSNRVec = [-20:5:5];
%TrainSet = 1e4;
trainSNR=0;
TrainSetVec = [1e2, 1e3, 1e4 1e5];
TestSet = 1e4;

N = 37;
numbers=primes(N-1);
Rset =  numbers(gcd(numbers,N)==1);

hiddenSize = 24;

for indexN = 1:length(TrainSetVec)

    TrainSet = TrainSetVec(indexN)

trainAns = [];
testAns  = [];

trainData2 = [];
testData2  = [];

trainData3 = [];
testData3 = [];

trainData4 = [];
testData4  = [];

trainData_RF=[];
testData_RF=[];

testData_AT=[];

parfor k=1:TrainSet
    
    R= 0; % initialization 
    R = Rset(randi(length(Rset),1));

    whoRU = randi(2,1);
    
    xx = zadoffChuSeq(R,N);
    % xx = xx / norm(xx);
    xn = awgn(xx,trainSNR,'measured');
    

    if whoRU == 1 % YES
        
    ansVec = [1 0];


      corrVec = (xcorr(xx,xn));

      [Un, Sn, Vn] = makeHankel(ifft(corrVec).');
      trainData2 = [trainData2 (diag(Sn))];
      trainData3 = [trainData3 abs(corrVec)];
      trainData4 = [trainData4 abs(ifft(corrVec))];
        
      a = abs(ifft(corrVec));
      featureVec = [a(1:N)', sum(a(1:N).^2)];
      trainData_RF = [trainData_RF; featureVec]; 
      
      
      trainAns = [trainAns transpose(ansVec)];
    else  % NO
    
    ansVec = [0 1];

      corrVec = (xcorr(xx,xn-xx));

      [Un, Sn, Vn] = makeHankel(ifft(corrVec).');
      trainData2 = [trainData2 (diag(Sn))];
      trainData3 = [trainData3 abs(corrVec)];
      trainData4 = [trainData4 abs(ifft(corrVec))];

      a = abs(ifft(corrVec));
      featureVec = [a(1:N)', sum(a(1:N).^2)];
      trainData_RF = [trainData_RF; featureVec]; % 행으로 추가

      trainAns = [trainAns transpose(ansVec)];
    end
end


trainLabels = categorical(vec2ind(trainAns)); % One-hot -> class labels
numTrees = 2;
RFModel = TreeBagger(numTrees, trainData_RF, trainLabels, 'Method', 'classification');

 layers2 = [
        featureInputLayer(size(trainData2,1))       
        fullyConnectedLayer(24)            
        %reluLayer
        leakyReluLayer
        fullyConnectedLayer(24)            
        %reluLayer
        leakyReluLayer
        fullyConnectedLayer(2)];

 layers3 = [
        featureInputLayer(size(trainData3,1))       
        fullyConnectedLayer(24)            
        %reluLayer
        leakyReluLayer
        fullyConnectedLayer(24)            
        %reluLayer
        leakyReluLayer
        fullyConnectedLayer(2)];


    % options = trainingOptions('adam', ...
    %     'MaxEpochs', MaxEpochs, ...
    %     'InitialLearnRate', InitialLearnRate, ...
    %     'MiniBatchSize', MiniBatchSize);
    options = trainingOptions('sgdm', ...
     'InitialLearnRate', 1e-3, ...
    'MaxEpochs', 30, ...
    'MiniBatchSize', TrainSet, ...
   'Shuffle', 'never', ...
   'Momentum',0.8,...
   'Verbose',false);

     net2 = trainnet(trainData2.', trainAns.', layers2,"mse", options);
     net3= trainnet(trainData3.', trainAns.', layers3,"mse",options);
     net4= trainnet(trainData4.', trainAns.', layers3,"mse", options);

     

parfor k=1:TestSet
    
    R= 0;
    R = Rset(randi(length(Rset),1));

    testSNR = trainSNR;

    whoRU = randi(2,1);
    
    xx = zadoffChuSeq(R,N);
    % xx = xx / norm(xx);

    xn = awgn(xx,trainSNR,'measured');
    if whoRU == 1 % YES
        
    ansVec = [1 0];


      corrVec = (xcorr(xx,xn));

      [Un, Sn, Vn] = makeHankel(ifft(corrVec).');
      testData2 = [testData2 (diag(Sn))];
      testData3 = [testData3 abs(corrVec)];
      testData4 = [testData4 abs(ifft(corrVec))];

      a = abs(ifft(corrVec));
      featureVec = [a(1:N)', sum(a(1:N).^2)];
      testData_RF = [testData_RF; featureVec];



      testAns = [testAns transpose(ansVec)];
    else  % NO
    
    ansVec = [0 1];

      corrVec = (xcorr(xx,xn-xx));

      [Un, Sn, Vn] = makeHankel(ifft(corrVec).');
      testData2 = [testData2 (diag(Sn))];
      testData3 = [testData3 abs(corrVec)];
      testData4 = [testData4 abs(ifft(corrVec))];

      a = abs(ifft(corrVec));
      featureVec = [a(1:N)', sum(a(1:N).^2)];
      testData_RF = [testData_RF; featureVec];
      
      testAns = [testAns transpose(ansVec)];
    end

    sortedCorr = sort(ifft(corrVec),'ascend');
    noiseLevel = mean(sortedCorr(1:round(0.3 * length(sortedCorr)))); % 잡음 평균 계산
    threshold = 3 * noiseLevel; % 임계값 설정 (α = 3)
    
    detected = max(ifft(corrVec)) > threshold; % Adaptive Threshold 방식 탐지 결과
    testData_AT = [testData_AT detected]; % 테스트 데이터 저장

end


%testResult2 = net2(testData2);
testResult2 = predict(net2, testData2.');
testResult2_mat = zeros(2,TestSet);

[ss ii] = max(testResult2.');
for k=1:TestSet
testResult2_mat(ii(k),k)=1;
end

%testResult3 = net3(testData3);
testResult3 = predict(net3, testData3.');
testResult3_mat = zeros(2,TestSet);

[ss ii] = max(testResult3.');
for k=1:TestSet
testResult3_mat(ii(k),k)=1;
end

testResult4 = predict(net4, testData4.');
testResult4_mat = zeros(2,TestSet);

[ss ii] = max(testResult4.');
for k=1:TestSet
testResult4_mat(ii(k),k)=1;
end

testResult_RF = str2double(predict(RFModel, testData_RF)); % 문자열 → double 변환

testResult_RF_f = zeros(2,TestSet);

for j = 1: TestSet

    if testResult_RF(j,:) == 1

        testResult_RF_f(1,j) = 1;
    else
        testResult_RF_f(2,j) =1;
    end
end

testResult_AT = zeros(2,TestSet);

for j = 1:TestSet
    
    if testData_AT(:,j) == 0

        testResult_AT(1,j)=1;
    else
        testResult_AT(2,j)=1;
    end

end


detectionRate2(indexN) = sum(sum(testResult2_mat.*testAns)) / TestSet;
detectionRate3(indexN) = sum(sum(testResult3_mat.*testAns)) / TestSet;
detectionRate4(indexN) = sum(sum(testResult4_mat.*testAns)) / TestSet;
detectionRate5(indexN) = sum(sum(testResult_RF_f.*testAns)) / TestSet;
detectionRate6(indexN) = sum(sum(testResult_AT.*testAns)) / TestSet;
end

%%
figure(1)
loglog(TrainSetVec, 1-detectionRate2,'r*-'); hold on
loglog(TrainSetVec, 1-detectionRate3, 'b-square'); hold on
loglog(TrainSetVec, 1-detectionRate4, 'g-+'); hold on
loglog(TrainSetVec, 1 - detectionRate5, 'm-o'); hold on
loglog(TrainSetVec, 1 - detectionRate6, 'k-^'); hold on
xlabel('Number of training dataset')
ylabel('Detection error rate')
legend('Proposed method','Deep learning method based on cross-correlation','Deep learning based on IFFT-processed cross-correlation', ...
    'Random forest method','Adaptive threshold method',...
    'location','best')
xticks(TrainSetVec)
%xlim([min(trainSNRVec) max(trainSNRVec)])
grid on
hold off
