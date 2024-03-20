#include <chrono>
#include <iostream>
#include <glog/logging.h>


class Timer {
public:
    Timer(const std::string& process) : process(process), start(std::chrono::high_resolution_clock::now()) {}

    ~Timer() {
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::micro> duration = end - start;

        std::string logMessage = "Time taken by " + process + ": " + std::to_string(duration.count()) + " microseconds";
        LOG(INFO) << logMessage;
    }

private:
    std::string process;
    std::chrono::time_point<std::chrono::high_resolution_clock> start;
};
