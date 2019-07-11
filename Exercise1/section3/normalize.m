function [ norm_X, avg, dev ] = normalize( d )

    dev = std(d);
    avg = mean(d);
    norm_X = (d - repmat(avg, size(d, 1), 1)) ./ repmat(dev, size(d, 1), 1);
end

