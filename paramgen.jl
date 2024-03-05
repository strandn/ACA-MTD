rank = [25, 30, 35, 40, 45, 50]
nc = [50]
ns = [5, 10, 15, 20]
jw = [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
io = open("params3.txt", "w")
for i in CartesianIndices((length(rank), length(nc), length(ns), length(jw)))
	write(io, "$(rank[i[1]]) $(nc[i[2]]) $(ns[i[3]]) $(jw[i[4]])\n")
end
close(io)
