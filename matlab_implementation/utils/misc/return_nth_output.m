function value = return_nth_output(N, fcn,varargin)
  [value{1:N}] = fcn(varargin{:});
  value = value{N};
end