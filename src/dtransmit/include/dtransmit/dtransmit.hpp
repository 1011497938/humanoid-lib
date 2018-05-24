// Created on: June 16, 2017
//     Author: Wenxing Mei <mwx37mwx@gmail.com>

#pragma once
#include <memory>
#include <vector>
#include <string>
#include <boost/asio/buffer.hpp>
#include <boost/asio/io_service.hpp>
#include <boost/asio/ip/udp.hpp>
#include <unordered_map>
#include <thread>
#include <functional>
#include <ros/ros.h>

// http://wiki.ros.org/roscpp/Overview/MessagesSerializationAndAdaptingTypes
#define UDPBUFFERSIZE 65535

namespace dtransmit {
class DTransmit {
    typedef int PORT;
public:
    explicit DTransmit(std::string address);
    ~DTransmit();

    template<typename ROSMSG>
    void addRosRecv(PORT port, std::function<void(ROSMSG &)> callback);

    template <typename ROSMSG>
    void sendRos(PORT port, ROSMSG &);

    void addRawRecv(PORT port, std::function<void(void*, std::size_t)> callback);

    void addRawRecvFiltered(PORT port, std::string remoteEndpoint, std::function<void(void*, std::size_t)> callback);

    void sendRaw(PORT port, const void* buffer, std::size_t size);

    void startService();

private:
    template<typename ReadHandler>
    void startRecv(PORT port, ReadHandler handler);
    void createSendSocket(PORT);
    void sendBuffer(boost::asio::ip::udp::socket*, const void* buffer, std::size_t size);

    // maybe use pair
    struct Foo {
        boost::asio::ip::udp::socket* socket;
        std::function<void(const boost::system::error_code& , std::size_t)> readHandler;
        uint8_t recvBuffer[UDPBUFFERSIZE];
        boost::asio::ip::udp::endpoint remoteEndpoint;

        Foo() {
        }

        // https://stackoverflow.com/a/39665940
        Foo(boost::asio::io_service& service, PORT port) {
            // construct the socket
            socket = new boost::asio::ip::udp::socket(service);

            // open it
            boost::asio::ip::udp::endpoint rx_endpoint_(boost::asio::ip::udp::v4(), port);
            boost::system::error_code error;
            socket->open(rx_endpoint_.protocol(), error);
            if(error) {
                ROS_ERROR("Can't open recv socket");
            } else {
               // then set it for reuse and bind it
                socket->set_option(boost::asio::ip::udp::socket::reuse_address(true));
                socket->bind(rx_endpoint_, error);
                if(error) {
                    ROS_ERROR("Can't bind recv socket");
                }
            }
        }

        ~Foo() {
        }
    };

    std::string m_broadcastAddress;
    boost::asio::io_service m_service;

    std::thread m_t;
    std::unordered_map<PORT, Foo> m_recvFoo;
    std::unordered_map<PORT, boost::asio::ip::udp::socket*> m_sendSockets;
};

template <typename ReadHandler>
void DTransmit::startRecv(PORT port, ReadHandler handler) {
    m_recvFoo[port].socket->async_receive_from(
        boost::asio::buffer(boost::asio::mutable_buffer((void*)&m_recvFoo[port].recvBuffer,
                                           sizeof(m_recvFoo[port].recvBuffer))), m_recvFoo[port].remoteEndpoint, handler
    );
}


template <typename ROSMSG>
void DTransmit::addRosRecv(PORT port, std::function<void(ROSMSG &)> callback) {
    ROS_INFO("Add Ros Recv on port: %d", port);
    using namespace boost::asio;
    if(m_recvFoo.count(port)) {
        ROS_ERROR("Error in addRosRecv: port %d exist!", port);
        return;
    }
    m_recvFoo[port] = Foo(m_service, port);

    m_recvFoo[port].readHandler = [=](const boost::system::error_code& error, std::size_t bytesRecved) {
        if(error) {
            ROS_ERROR("Error in RosRecv: %s", error.message().c_str());
        } else {
            try {
                ROSMSG msg;

                ros::serialization::IStream stream((uint8_t*)m_recvFoo[port].recvBuffer, bytesRecved);
                ros::serialization::Serializer<ROSMSG>::read(stream, msg);
                // client callback
                callback(msg);
            } catch (std::exception& e) {
                ROS_ERROR("%s", e.what());
            }
        }

        startRecv(port, m_recvFoo[port].readHandler);
    };
    startRecv(port, m_recvFoo[port].readHandler);
}


template <typename ROSMSG>
void DTransmit::sendRos(PORT port, ROSMSG &rosmsg) {
    try {
        // serialize rosmsg
        uint32_t serial_size = ros::serialization::serializationLength(rosmsg);
        //std::unique_ptr<uint8_t> buffer(new uint8_t[serial_size]);
        auto buffer = new uint8_t[serial_size];

        ros::serialization::OStream stream(buffer, serial_size);
        ros::serialization::serialize(stream, rosmsg);

        sendRaw(port, buffer, serial_size);
        //!? leak memory on exception
        delete[] buffer;
    } catch(std::exception& e) {
        ROS_ERROR("%s", e.what());
    }
}

} // namespace dtransmit
