clc
clear
close all

%%=================================code====================================
%% Network size
input_nodes = 2;
hidden_layer_neurons = 2;
output_layer_neurons = 2;

%% training patterns
s = [0.25 0.25; 0.5 0.5; 0.75 0.25; 0.75 0.75];
% desired output
t = s;

%% add bias to s
s_bias = [s(1,:) 1; s(2,:) 1; s(3,:) 1; s(4,:) 1];

%% length of training patterns
s_length = length(s);

%% initialization weights
W_1 = rands(input_nodes + 1,hidden_layer_neurons);
W_2 = rands(hidden_layer_neurons + 1,output_layer_neurons);

%% matrix for update calculation
W_1_upd = zeros(input_nodes + 1,hidden_layer_neurons);
W_2_upd = zeros(hidden_layer_neurons + 1,output_layer_neurons);

%% Initialization of parameters 
alpha = 0.1;
landa = 0.7;
epoch = 1;
error_epoch(epoch) = 1;
mse_error_epoch(epoch) = 1;

%% train network
while(mse_error_epoch(epoch) > 0.0001 && epoch < 10000)
    % reset error of epoch
    square_error_total =0;
    % reset counter of epoch
    c = 0;
    % next epoch
    epoch = epoch + 1;
    for i_samples = 1:s_length
        % calculate input of hidden layer
        h_in = s_bias(i_samples,:) * W_1;
        % calculate output of hidden layer
        H = 1 ./ (1 + exp(-1 * h_in));
        % add bias
        H_bias = [H 1];
        % calculate input of output layer
        y_in = H_bias * W_2;
        % calculate output of Network
        Y(i_samples,:,epoch) = 1 ./ (1+exp(-1 * y_in));
        % calculate square error
        square_error_sample = (t(i_samples,:)-Y(i_samples,:,epoch)).^2;
        square_error_total = square_error_total + sum(square_error_sample);
        % calculate delta
        delta_y = (t(i_samples,:)-Y(i_samples,:,epoch)) .* (0.5*Y(i_samples,:,epoch)) .* (1-Y(i_samples,:,epoch));
        delta_h = (0.5*(H_bias(1,1:2))) .* (1-H_bias(1,1:2)) .* (sum(delta_y .* W_2));
        % calculate w updates
        W_2_upd = alpha * H_bias' * delta_y + landa * W_2_upd;
        W_1_upd = alpha * s_bias(i_samples,:)' * delta_h + landa * W_1_upd;
        % updating weights
        W_2 = W_2 + W_2_upd;
        W_1 = W_1 + W_1_upd;
    end
    % save error each epoch
    error_epoch(epoch)=square_error_total;
    mse_error_epoch(epoch)=sqrt(square_error_total/8);
    % decrease alpha
    c = c + 1; % counter + 1
    if c == 50 % if counter == 500 then decrease alpha
        alpha = alpha / 2;
    end
end

%% plot square error
figure
mse_error_epoch(1) = [];
plot(mse_error_epoch);
title('mse error');

%% plot patterns & output
figure
plot(s(:,1),s(:,2),'b*','markersize',10);
axis([0 1 0 1]);
grid on
hold on
for i_epoch = 1:50:epoch
    plot(Y(:,1,i_epoch),Y(:,2,i_epoch),'r+','markersize',5);
    hold on
    pause(0.001)
end
title('result of training');
legend('training patterns','network output');
hold off

%% part2_2 : test
% test samples
x_test = rand(10,2);
% lenght test samples
length_test = length(x_test);
% start
for i_test_sample = 1:length_test
    input = [x_test(i_test_sample,:) 1];
    for iteration = 1:100
        % calculate input of hidden layer
        h_in = input * W_1;
        % calculate output of hidden layer
        H = 1 ./ (1 + exp(-1 * h_in));
        % add bias
        H_bias = [H 1];
        % calculate input of output layer
        y_in = H_bias * W_2;
        % calculate output of Network
        Y_test(i_test_sample,:,iteration) = 1 ./ (1+exp(-1 * y_in));
        input = [Y_test(i_test_sample,:,iteration) 1];
    end
end

% plot test sample & output
figure
hold on
for i_test_sample = 1:length_test
    plot(x_test(i_test_sample,1),x_test(i_test_sample,2),'.r','markersize',2);
    title('result of test');
end
hold on
for i_test_sample = 1:length_test
    for count = 1:iteration
        plot(Y_test(i_test_sample,1,count),Y_test(i_test_sample,2,count),'.b' , 'MarkerSize' , 2);
        hold on
    end
end
grid on
axis([0 1 0 1])
legend('training patterns','test sample','output')

%% part2_3 : plot boundaries
% plot training patterns
plot(s(:,1),s(:,2),'r*','markersize',10);
hold on
% Specify the axis
x = 0 : 0.001 : 1;
% compute each boundry
l1 = (W_1(1,1)/W_1(2,1)).*x + (W_1(3,1)/W_1(2,1));
l2 = (W_1(1,2)/W_1(2,2)).*x + (W_1(3,2)/W_1(2,2));
plot(x,l1,'r--')
hold on
plot(x,l2,'r--')
hold on
l3 = (W_2(1,1)/W_2(2,1)).*x + (W_2(3,1)/W_1(2,1));
l4 = (W_2(1,2)/W_2(2,2)).*x + (W_2(3,2)/W_1(2,2));
plot(x,l3,'m--')
hold on
plot(x,l4,'m--')

%%=================================END=====================================
    



