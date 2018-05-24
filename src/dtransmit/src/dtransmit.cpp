#include "dtransmit/dtransmit.hpp"
#include <algorithm>

using namespace std;
using namespace boost::asio;

namespace dtransmit {
DTransmit::DTransmit(string address)
    : m_broadcastAddress(address), m_service()
{
    std::cout << "New DTransmit on " << address << std::endl;
}

DTransmit::~DTransmit() {
    m_service.stop();
    for(auto& p : m_sendSockets) {
        ip::udp::socket* s = p.second;
        if(s) {
            if(s->is_open())
                s->close();
            delete s;
        }
    }

    for(auto& p : m_recvFoo) {
        ip::udp::socket* s = p.second.socket;
        if(s) {
            if(s->is_open())
                s->close();

            delete s;
        }
    }

    m_service.stop();
    if(m_t.joinable())
        m_t.join();
}

void DTransmit::startService() {
    m_t = std::thread([&]() {
        m_service.reset();
        m_service.run();
    });
}

void DTransmit::createSendSocket(PORT port) {
    using namespace boost::asio;
    ip::udp::endpoint broadcastEndpoint(ip::address::from_string(m_broadcastAddress), port);
    m_sendSockets[port] = new ip::udp::socket(m_service, ip::udp::v4());
    m_sendSockets[port]->set_option(socket_base::broadcast(true));

    boost::system::error_code ec;
    m_sendSockets[port]->connect(broadcastEndpoint, ec);
    if(ec) {
        ROS_ERROR("DTransmit create sendRos socket error: %s", ec.message().c_str());
    }
}

void DTransmit::sendBuffer(boost::asio::ip::udp::socket* socket, const void* buffer, std::size_t size) {
    boost::system::error_code ec;
    socket->send(boost::asio::buffer(buffer, size), 0, ec);
    if(ec) {
        // ROS_WARN("DTransmit can't send Ros buffer: %s", ec.message().c_str());
    }
}

void DTransmit::addRawRecv(PORT port, std::function<void(void*, std::size_t)> callback) {
    if(m_recvFoo.count(port)) {
        ROS_ERROR("Error in addRawRecv: port %d exist!", port);
        return;
    }
    m_recvFoo[port] = Foo(m_service, port);

    m_recvFoo[port].readHandler = [=](const boost::system::error_code& error, std::size_t bytesRecved) {
        if(error) {
            ROS_ERROR("Error in RosRecv: %s", error.message().c_str());
        } else {
            callback(m_recvFoo[port].recvBuffer, bytesRecved);
        }

        startRecv(port, m_recvFoo[port].readHandler);
    };
    startRecv(port, m_recvFoo[port].readHandler);
}

void DTransmit::addRawRecvFiltered(PORT port,
                                   std::string remoteEndpoint,
                                   std::function<void(void *, std::size_t)> callback) {
    if(m_recvFoo.count(port)) {
        ROS_ERROR("Error in addRawRecv: port %d exist!", port);
        return;
    }
    m_recvFoo[port] = Foo(m_service, port);

    m_recvFoo[port].readHandler = [=](const boost::system::error_code& error, std::size_t bytesRecved) {
        if(error) {
            ROS_ERROR("Error in RosRecv: %s", error.message().c_str());
        } else if(m_recvFoo[port].remoteEndpoint.address().to_string() != remoteEndpoint) {
            // ROS_WARN("Someone else [%s] is broadcasting on this port [%d], packet filtered.", m_recvFoo[port].remoteEndpoint.address().to_string().c_str(), port);
        } else {
            callback(m_recvFoo[port].recvBuffer, bytesRecved);
        }

        startRecv(port, m_recvFoo[port].readHandler);
    };
    startRecv(port, m_recvFoo[port].readHandler);

}

void DTransmit::sendRaw(PORT port, const void *buffer, std::size_t size) {
    if(!m_sendSockets.count(port)) {
        createSendSocket(port);
    }
    sendBuffer(m_sendSockets[port], buffer, size);
}


}
