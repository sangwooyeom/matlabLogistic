function main()
  funs = utils;
  lambda = {0.001, 0.01, 0.1, 1.0};
  endPoint = zeros(1,4);
  hyperparameters = struct('learning_rate', lambda, 'num_iterations', 1000);
  train_type = struct('all', 1, 'normal', 0, 'small', 0);

  logging = zeros(getfield(hyperparameters,'num_iterations') * length(lambda), 6);


  for i = 1:length(lambda)
    start_num = (getfield(hyperparameters,'num_iterations')*(i-1)+1);
    last_num = getfield(hyperparameters,'num_iterations')*i;
    [logging(start_num:last_num,:)] = funs.run_logistic_regression(train_type, hyperparameters(i));
    end_num = start_num + max(logging(start_num:last_num,6)) -1;
    endPoint(i) = end_num;
  end

  titleArr = {'training cross entropy','training accuracy','valid cross entropy','valid accuracy'};
  % figure
  % for i = 2:5
  %   subplot(2,2,i-1)
  %   plot(logging(1:endPoint(1),i),'DisplayName','learning rate = 0.001');
  %   hold on
  %   plot(logging(1001:endPoint(2),i),'DisplayName','learning rate = 0.01');
  %   hold on
  %   plot(logging(2001:endPoint(3),i),'DisplayName','learning rate = 0.1');
  %   hold on
  %   plot(logging(3001:endPoint(4),i),'DisplayName','learning rate = 1.0');
  %   hold off
  %   legend('show')
  %   title(titleArr(i-1))
  % end
  % suptitle('evaluate logistic regression')

  % when lambda = 4 valid result is best

  num_runs = 10; %number of runs
  test_error_rate = zeros(1, num_runs);

  test_inputs = readNPY('mnist_test/test_inputs.npy');
  test_targets = readNPY('mnist_test/test_targets.npy');
  szt = size(test_targets);

  for i = 1:num_runs
    [logging_final, weights_final] = funs.run_logistic_regression(train_type, hyperparameters(4));
    test_res = funs.logistic_predict(weights_final, test_inputs);

    % threshold : 0.5
    test_res(test_res>0.5) = 1;
    test_res(test_res<=0.5) = 0;

    % test error rate
    test_error_rate(i) = 1 - sum(test_res == test_targets) / szt(1);
    fprintf('%.0d%% ',test_error_rate(i) * 100);
  end
  fprintf('\n');
end
