// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.0;

contract CheckWeights {
    struct Client {
        uint federatedCid;
        mapping(uint => address) roundWeightsReference;
    }

    Client[] public client;
    uint[] public blacklist;

    address private owner;

    constructor() {
        owner = msg.sender;
    }

    modifier onlyOwner() {
        require(msg.sender == owner, "Only the owner can call this function.");
        _;
    }

    function getOwner() public view returns (address) {
        return owner;
    }

    event WroteWeightsOfRoundForClient(
        address weights,
        uint round,
        uint federatedCid
    );

    function storeWeightOfRoundForClient(
        string memory _weights,
        uint _round,
        uint _federatedCid
    ) public {
        bool clientExists = checkClientExist(_federatedCid);
        if (clientExists) {
            Client storage clientItem = client[findClientIndex(_federatedCid)];
            clientItem.roundWeightsReference[_round] = address(bytes20(bytes(_weights)));
        } else {
            //solidity not support push of item that has mapping => first add empty item and after set its properties
            uint lastItem = client.length;
            client.push();
            Client storage newClient = client[lastItem];
            newClient.federatedCid = _federatedCid;
            newClient.roundWeightsReference[_round] = address(bytes20(bytes(_weights)));
        }

        emit WroteWeightsOfRoundForClient(address(bytes20(bytes(_weights))), _round, _federatedCid); //throw wrote event for trigger check  of weights
    }

    function findClientIndex(uint _federatedCid) internal view returns (uint) {
        for (uint i = 0; i < client.length; i++) {
            if (client[i].federatedCid == _federatedCid) {
                return i;
            }
        }
        revert("Client not found");
    }

    function checkClientExist(uint _federatedCid) internal view returns (bool) {
        for (uint i = 0; i < client.length; i++) {
            if (client[i].federatedCid == _federatedCid) {
                return true;
            }
        }
        return false;
    }

    function addToBlacklist(uint _federatedCid) public {
        blacklist.push(_federatedCid);
    }

    function getBlacklist() public view onlyOwner returns (uint[] memory) {
        return blacklist;
    }
}
