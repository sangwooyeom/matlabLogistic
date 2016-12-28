weights = rand(1,28*28+1)';
data = readNPY('mnist_train/train_inputs.npy');
targets = readNPY('mnist_train/train_targets.npy');

[N, M] = size(data);
w = weights(1:end-1);
bias = weights(end);

z = zeros(N,1);
y = zeros(N,1);

for i = 1:N
  z(i) = bias + dot(data(i,:),w);
  y(i) = my.sigmoid(z(i));
end

df = zeros(M+1,1);
for i = 1:M
  df(i) = dot(y - targets,data(:,i));
end


  df(end) = dot(y, -exp(-z)) + dot(ones(N,1), 1-targets);

  fs = my.evaluate(targets,y);
