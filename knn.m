function valid_labels = knn(k, train_data, train_labels, valid_data)
  % Uses the supplied training inputs and labels to make
  % predictions for validation data using the K-nearest neighbours
  % algorithm.
  %
  % Note: N_TRAIN is the number of training examples,
  %       N_VALID is the number of validation examples,
  %       and M is the number of features per example.
  %
  % Inputs:
  %     k:            The number of neighbours to use for classification
  %                   of a validation example.
  %     train_data:   The N_TRAIN x M array of training
  %                   data.
  %     train_labels: The N_TRAIN x 1 vector of training labels
  %                   corresponding to the examples in train_data
  %                   (must be binary).
  %     valid_data:   The N_VALID x M array of data to
  %                   predict classes for.
  %
  % Outputs:
  %     valid_labels: The N_VALID x 1 vector of predicted labels
  %                   for the validation data.
  trainSize = size(train_data);
  validSize = size(valid_data);
  distMat = pdist2(train_data,valid_data);

  valid_labels = zeros(validSize(1),1);
  for i = 1:validSize(1)
    sorted = sortrows([distMat(:, i), train_labels],1);
    tmp = sum(sorted(1:k,2))/k;
    if tmp > 0.5
      valid_labels(i) = 1;
    end
  end
end
