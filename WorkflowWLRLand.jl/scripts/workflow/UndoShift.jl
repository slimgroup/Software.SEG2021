function UndoShift(AShot,datainfo)
	### Input:
	### AShot: one shot gather (nt,nrx,nry) with padding zeros
	### datainfo: a Dictionary contains info for one shot gather
	### Output:
	### A shot gather w/o linear shift
	nrx = datainfo["nrx"];
	nry = datainfo["nry"];
	Rgrid = datainfo["Rgrid"];
	dt = datainfo["dt"];

	vel = 1415; #velocity sand(dry) 0.2-1 Km/s, sand(water saturated 1.5-2 Km/s)

	# find the center position of receiver = source position
	CentX = (nrx+1)/2;
	CentY = (nry+1)/2;

	# calculate the shift grid for each receiver and save them in the matrix shiftT
	a = 1:nrx; 
	b = 1:nry;
	row = reshape(repeat(a,outer=nry),nrx,nry);
	col = reshape(repeat(b,inner=nry),nrx,nry);
	shiftT = round.(sqrt.((row .-CentX).^2 .+ (col .-CentY).^2) .*(Rgrid/vel/(dt*1e-3)));

	nt1 =  datainfo["nt"] + 300 + 100;
	Shift_Back_Data = zeros(size(AShot));
	for i = 1:nrx
	    for j = 1:nry
		Shift_Back_Data[Int(shiftT[i,j]+1):nt1,i,j] = AShot[1:Int(nt1-shiftT[i,j]),i,j];
	    end
	end

	return Shift_Back_Data[300+1:300+datainfo["nt"],:,:]
end
