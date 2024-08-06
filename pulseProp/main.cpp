#include <iostream>
#include "pulseProp.h"

int main(int, char**) {
    std::string helloJim = generateHelloString("Jim");
    std::cout << helloJim << std::endl;

    std::cin.get();
    return 0;
}