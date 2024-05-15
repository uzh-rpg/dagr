class SpiralOut{
protected:
    unsigned layer;
    unsigned leg;
public:
    int x, y; //read these as output from next, do not modify.
    __device__ SpiralOut():layer(1),leg(0),x(0),y(0){}
    __device__ void goNext(){
        switch(leg){
        case 0: ++x; if(x  == layer)  ++leg;                break;
        case 1: ++y; if(y  == layer)  ++leg;                break;
        case 2: --x; if(-x == layer)  ++leg;                break;
        case 3: --y; if(-y == layer){ leg = 0; ++layer; }   break;
        }
    }
};