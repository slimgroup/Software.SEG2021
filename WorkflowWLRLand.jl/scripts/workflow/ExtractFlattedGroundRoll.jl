using DSP
function ExtractFlattedGroundRoll(AShot,datainfo)
    ### Input:
	### AShot: one shot gather (nt,nrx,nry)
	### datainfo: a Dictionary contains info for one shot gather
	### Output:
	### A shot gather with only the flatted ground roll estimatation
	nrx = datainfo["nrx"]; #number of receiver x
	nry = datainfo["nry"]; #number of receiver y
	Rgrid = datainfo["Rgrid"]; #receiver grid space
	dt = datainfo["dt"]; # number of time
	# Pad zeros in time domain for the shot data
	AShot = [zeros(300,nrx,nry);AShot;zeros(100,nrx,nry)]

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

	Shift_Data = zeros(size(AShot));
	nt1 =  datainfo["nt"] + 300 + 100;
	for i = 1:nrx
	    for j = 1:nry
		Shift_Data[1:Int(nt1-shiftT[i,j]),i,j] = AShot[Int(shiftT[i,j]+1):nt1,i,j];
	    end
	end

	# define a filter in time domain to mute the signal besides groundroll
	A = zeros(nt1,1);
	band = 10;
	low = 230;
	high = 540;

	A[low+band+1:high-band-1] .= DSP.Windows.tukey(high-band-1-(low+band+1)+1,0.6;padding=0,zerophase=false)

	Shift_taper_Data = zeros(size(Shift_Data));
	for i = 1:nrx
	    for j = 1:nry
		Shift_taper_Data[:,i,j] = Shift_Data[:,i,j] .* A;
	    end
	end
	return Shift_taper_Data;
end
