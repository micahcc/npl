
set.seed(13)
tall.data = matrix(rnorm(1024), nrow=128, ncol=8)
wide.data = matrix(rnorm(1024), nrow=8, ncol=128)

write.json = function(mat, filename)
{
	cat("{\"type\": \"double\",\n\"size\": [",ncol(mat), ",1,1,", nrow(mat), 
		"],\n", file=filename, append=F, sep='')
	cat("\"values\":\n[", file=filename, append=T)
	for(cc in seq(ncol(mat))) {
		if(cc != 1) {
			cat(",\n", file= filename, append=T, sep='')
		}
		cat("[[[", file= filename, append=T)
		cat(mat[,cc], file=filename, sep=', ', append=T)
		cat("]]]", file= filename, append=T)
	}	
	cat("]\n}", file= filename, append=T)
}

write.json.split = function(mat, rcount, ccount, base)
{
	if(nrow(mat)%%rcount != 0) {
		print("Error matrix not divisible by rcount")
	}
	if(ncol(mat)%%ccount != 0) {
		print("Error matrix not divisible by ccount")
	}

	crows = nrow(mat)/rcount
	ccols = ncol(mat)/ccount
	for(rr in seq(rcount)) {
		for(cc in seq(ccount)) {
			fn = paste(base, "c", cc, "r", rr, ".json", sep="")
			write.json(mat[(crows*(rr-1)+1):(crows*rr), (ccols*(cc-1)+1):(ccols*cc)], fn)
		}
	}
}

write.json(tall.data, 'talldata.json')
write.json(wide.data, 'widedata.json')

# split into four
write.json.split(tall.data, 2, 2, "talldata_") 
write.json.split(wide.data, 2, 2, "widedata_") 
