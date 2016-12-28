% function list
% 1. sigmoid
% 2. plot_digits
% 3. show_pane
% 4. evaluate
% 5. check_grad
% 6. run_check_grad
% 7. logistic_predict
% 8. run_logistic_regression
% 9. print_log

function funs = utils
  funs.sigmoid=@sigmoid;
  funs.plot_digits=@plot_digits;
  funs.show_pane=@show_pane;
  funs.evaluate=@evaluate;
  funs.check_grad=@check_grad;
  funs.run_check_grad=@run_check_grad;
  funs.logistic_predict=@logistic_predict;
  funs.run_logistic_regression=@run_logistic_regression;
  funs.print_log=@print_log;
end

% 1. def sigmoid function
function result = sigmoid(x)
  %Inputs:
    %  x: Either a row vector or a column vector.
  result = 1 / (1.0 + exp(-x));
end

% 2. plot_digits

function plot_digits(digit_array)
  % Visualizes each example in digit_array.
  %
  % Note: N is the number of examples
  %       and M is the number of features per example.
  %
  % Inputs:
  %     digits: N x M array of pixel intensities.
  arrSize = size(digit_array);
  CLASS_EXAMPLES_PER_PANE = 5;
  examples_per_class = arrSize(1)/2;
  num_panes = ceil(examples_per_class/CLASS_EXAMPLES_PER_PANE);

  for pane = 1:num_panes
    fprintf('Displaying pane %d / %d\n', pane, num_panes);
    start_index = 1 +  (pane-1) * CLASS_EXAMPLES_PER_PANE * 2;
    end_index = pane * CLASS_EXAMPLES_PER_PANE * 2;
    show_pane(digit_array, start_index, end_index);
  end
end

% 3. show_pane
function show_pane(digit_array, start_index, end_index)
  % Displays two rows of digits on the screen.
  width = 28;
  height = 28;
  figure
  for i = start_index:end_index
    dblImage = im2double(reshape(digit_array(i, :),[width,height])');
    subplot(2,5,ceil(rem(i-0.1,10)));
    imshow(dblImage,[]);
  end
end


% 4. evaluate
function [ce, p] = evaluate(targets, y_hat)
  % targets : true class of data and y_hat : predicted class
  % return cross entropy and accuracy percents

  % cross entropy
  ce = -dot(targets, log(y_hat)) - dot(1-targets, log(1-y_hat));

  % accuracy percents
  ts = size(targets);
  correctArr = ones(ts(1),1);

  correctArr = correctArr .* (abs(targets - y_hat) > 0.5);
  p = 1-sum(correctArr)/ts(1);
end

% 5. check_grad
function d  = check_grad(func, X, epsilon, data, targets, hyperparameters)
  % where X is the argument and epsilon is the small perturbation used for the finite
  % differences. and the P1, P2, ... are optional additional parameters which
  % get passed to f. The function f should be of the type
  %
  % (fX, dfX) = func(X, P1, P2, ...)
  %
  % where fX is the function value and dfX is a vector of partial derivatives.

  [y, dy, z] = func(X, data, targets, hyperparameters);

  dh = zeros(length(X),1);

  for j = 1:length(X)
      dx = zeros(length(X),1);
      dx(j) = epsilon;
      y2 = func(X+dx, data, targets, hyperparameters);
      dx = -dx;
      y1 = func(X+dx, data, targets, hyperparameters);
      dh(j) = (y2 - y1)/(2*epsilon);
  end
  d = norm(dh-dy) / norm(dh+dy);
end

% 6. run_check_grad
function run_check_grad(hyperparameters)

  num_examples = 7;
  num_dimensions = 9;

  weights = rand(num_dimensions+1, 1);
  data    = rand(num_examples, num_dimensions);
  targets = double(rand(num_examples, 1) > 0.5);

  diff = check_grad(@logistic,weights, 0.001,data,targets,hyperparameters);

  fprintf ('diff = %d\n', diff);
end

% 7. logistic_predict
function y = logistic_predict(weights, data)
  [N, M] = size(data);
  w = weights(1:end-1);
  bias = weights(end);

  z = zeros(N,1);
  y = zeros(N,1);

  for i = 1:N
    z(i) = bias + dot(w,data(i,:));
    y(i) = sigmoid(z(i));
  end
end

% 8. run_logistic_regression
function [logging, weights] = run_logistic_regression(type, hyperparameters)
  % load data
  if getfield(type,'normal') == 1
    train_inputs = readNPY('mnist_train/train_inputs.npy');
    train_targets = readNPY('mnist_train/train_targets.npy');
  elseif getfield(type,'small') == 1
    train_inputs = readNPY('mnist_train_small/train_inputs.npy');
    train_targets = [0 0 0 0 0 1 1 1 1 1]';
  elseif getfield(type,'all') == 1
    train_inputs = readNPY('mnist_train/train_inputs.npy');
    train_targets = readNPY('mnist_train/train_targets.npy');
    train_inputs_small = readNPY('mnist_train_small/train_inputs.npy');
    train_targets_small = [0 0 0 0 0 1 1 1 1 1]';

    train_inputs = vertcat(train_inputs, train_inputs_small);
    train_targets = vertcat(train_targets, train_targets_small);
   else
       ;
   end

   valid_inputs = readNPY('mnist_valid/valid_inputs.npy');
   valid_targets = readNPY('mnist_valid/valid_targets.npy');


  % N is number of examples; M is the number of features per example.
  [N, M] = size(train_inputs);

  % Logistic regression weights
  weights = rand(M+1,1) - 0.5;


  % run_check_grad(hyperparameters);

  logging = zeros( getfield(hyperparameters,'num_iterations'), 6) ;
  lastT = 1;
  for t = 1:getfield(hyperparameters,'num_iterations')
    if  rem(t,getfield(hyperparameters,'num_iterations')) > 12 & abs(logging(t-7,2) - logging(t-6,2))/N < 0.0005
      break;
    else
      % Find the negative log likelihood and its derivatives w.r.t. the weights.
      [f, df, y_hat] = logistic(weights, train_inputs, train_targets, hyperparameters);

      % Evaluate the prediction.
      [train_ce, train_frac] = evaluate(train_targets, y_hat);

      if isnan(f) || isinf(f)
        disp( 'ValueError("nan/inf error")' );
        logging(t,:) = [nan, nan, nan, nan, nan, nan];
        continue;
      end

      % update parameters
      weights = weights - (getfield(hyperparameters,'learning_rate') * df / N);

      % Make a prediction on the valid_inputs.
      valid_hat = logistic_predict(weights, valid_inputs);

      % Evaluate the prediction.
      [valid_ce, valid_frac] = evaluate(valid_targets, valid_hat);
      logging(t,:) = [f / N, train_ce, train_frac*100, valid_ce, valid_frac*100, t];
      lastT = t;
    end
  end
  for t = (lastT+1):getfield(hyperparameters,'num_iterations')
    logging(t,:) = [0,0,0,0,0,lastT];
  end
end

% 9. print log
function print_log(log)
  for i = 1:length(log)
    fprintf('ITERATION: %d  TRAIN NLOGL: %.2f TRAIN CE: %.6f ', i, log(1), log(2));
    fprintf('TRAIN FRAC: %.2f  VALID CE: %.6f  VALID FRAC: %.2f \n', log(3), log(4),log(5));
  end
end
