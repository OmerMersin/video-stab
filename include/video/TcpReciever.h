#include <atomic>
#include <cstdint>
#include <thread>
#include <cstdio>  // for sscanf
#include <string>

namespace vs {

class TcpReciever
{
public:
    explicit TcpReciever(uint16_t port);
    ~TcpReciever();

    bool start();                       ///< start listener (returns false on error)
    void stop();                        ///< stop and join thread

    /** Get most-recent coordinates that arrived over TCP.
        @return true if a new pair was returned, false otherwise.              */
    bool tryGetLatest(int& outX, int& outY);

private:
    void              listenLoop();

    uint16_t          port_;
    int               listenFd_{-1};
    std::thread       thread_;
    std::atomic<bool> running_{false};

    std::atomic<int>  latestX_{-1};
    std::atomic<int>  latestY_{-1};
};

} // namespace vs