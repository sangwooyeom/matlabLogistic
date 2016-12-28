function main_regularized()
  funs = utils;
  regularized = {0.001, 0.01, 0.1, 1.0};
  lambda = {0.05, 0.25, 1.0};
  endPoint = zeros(1,4);

  train_type = struct('all', 1, 'normal', 0, 'small', 0);

  evalParameters = zeros(length(lambda),length(regularized));
  for i = 1:length(lambda)
    logging = zeros(1000 * length(regularized), 6);
    for j = 1:length(regularized)
      hyperparameters = struct('learning_rate', lambda(i), 'num_iterations', 1000, 'weight_regularization', regularized(j) );
      start_num = (getfield(hyperparameters,'num_iterations')*(j-1)+1);
      last_num = getfield(hyperparameters,'num_iterations')*j;
      [logging(start_num:last_num,:)] = funs.run_logistic_regression(train_type, hyperparameters);
      end_num = start_num + max(logging(start_num:last_num,6)) -1;
      evalParameters(i,j) = mean(logging((end_num-10):end_num,2))+2*mean(logging((end_num-10):end_num,4));
      endPoint(j) = end_num;
    end

    titleArr = {'training cross entropy','training error','valid cross entropy','valid error'};
    figure
    for j = 2:5
      subplot(2,2,j-1)
      if rem(j,2) == 1
        plot(100 - logging(1:endPoint(1),j),'DisplayName','regularized = 0.001');
        hold on
        plot(100 - logging(1001:endPoint(2),j),'DisplayName','regularized = 0.01');
        hold on
        plot(100 - logging(2001:endPoint(3),j),'DisplayName','regularized = 0.1');
        hold on
        plot(100 - logging(3001:endPoint(4),j),'DisplayName','regularized = 1.0');
      else
        plot(logging(1:endPoint(1),j),'DisplayName','regularized = 0.001');
        hold on
        plot(logging(1001:endPoint(2),j),'DisplayName','regularized = 0.01');
        hold on
        plot(logging(2001:endPoint(3),j),'DisplayName','regularized = 0.1');
        hold on
        plot(logging(3001:endPoint(4),j),'DisplayName','regularized = 1.0');
      end
      hold off
      legend('show')
      title(titleArr(j-1))
    end
    suptitle(strcat('evaluate regularized logistic regression learning rate = ',num2str(cell2mat(lambda(i)))));
  end

  [minV,index] = min(evalParameters(:));
  selected_lambda = lambda(rem(index+2,length(lambda))+1);
  selected_regularized = regularized(ceil(index/length(lambda)));
  % lambda = 1, regularized = 0.01

  num_runs = 10; %number of runs
  test_error_rate = zeros(1, num_runs);

  test_inputs = readNPY('mnist_test/test_inputs.npy');
  test_targets = readNPY('mnist_test/test_targets.npy');
  szt = size(test_targets);

  for i = 1:num_runs
    hyperparameters = struct('learning_rate', selected_lambda,'num_iterations', 1000, 'weight_regularization', selected_regularized);
    [logging_final, weights_final] = funs.run_logistic_regression(train_type,hyperparameters);
    test_res = funs.logistic_predict(weights_final, test_inputs);

    % threshold : 0.5
    test_res(test_res>0.5) = 1;
    test_res(test_res<=0.5) = 0;

    % test error rate
    test_error_rate(i) = 1 - sum(test_res == test_targets) / szt(1);
    fprintf('%.0f%% ',test_error_rate(i) * 100);
  end
  fprintf('\n');

end
