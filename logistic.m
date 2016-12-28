function [f, df,y] = logistic(weights, data, targets, hyperparameters)
  % Calculate negative log likelihood and its derivatives with respect to weights.
  % Also return the predictions.
  %
  % Note: N is the number of examples and
  %       M is the number of features per example.
  %
  % Inputs:
  %     weights:    (M+1) x 1 vector of weights, where the last element
  %                 corresponds to bias (intercepts).
  %     data:       N x M data matrix where each row corresponds
  %                 to one data point.
  %     targets:    N x 1 vector of targets class probabilities.
  %     hyperparameters: The hyperparameters dictionary.
  %
  % Outputs:
  %     f:       The sum of the loss over all data points. This is the objective that we want to minimize.
  %     df:      (M+1) x 1 vector of accumulative derivative of f w.r.t. weights, i.e. don't need to average over number of sample
  %     y:       N x 1 vector of probabilities.
  my = utils;

  [N, M] = size(data);
  w = weights(1:end-1);
  bias = weights(end);

  z = zeros(N,1);
  y = zeros(N,1);
  df = zeros(M+1,1);

  for i = 1:N
    z(i) = bias + dot(data(i,:),w);
    y(i) = my.sigmoid(z(i));
  end

  if any(isfield(hyperparameters,'weight_regularization'))
    alpha = getfield(hyperparameters,'weight_regularization');
    % f is the loss function plus lambda/2 * sigma (wi^2) : lw penalized
    f = my.evaluate(targets,y)+ 0.5* alpha * (dot(weights',weights));
    for i = 1:M
      df(i) = dot(y - targets,data(:,i))+ alpha * weights(i);
    end
  else
    for i = 1:M
      df(i) = dot(y - targets,data(:,i));
    end
    f = my.evaluate(targets,y);
  end

end
