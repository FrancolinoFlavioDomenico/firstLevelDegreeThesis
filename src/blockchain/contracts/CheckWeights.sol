// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.0;
pragma abicoder v2;

contract CheckWeights {
    mapping(uint => string[]) roundWeightsReference;

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
        string weightsHash,
        uint round,
        uint federatedCid
    );

    function addRoundWeightsReference(
        string memory _weightsHash,
        uint _round,
        uint _federatedCid
    ) public {
        // roundWeightsReference[_round] = _weightsHash;
        roundWeightsReference[_federatedCid].push(_weightsHash);
        emit WroteWeightsOfRoundForClient(roundWeightsReference[_federatedCid][_round], _round, _federatedCid);//TODO fix event value for weight reference
    }




    // function findClientIndex(uint _federatedCid) internal view returns (uint) {
    //     for (uint i = 0; i < clients.length; i++) {
    //         if (clients[i].federatedCid == _federatedCid) {
    //             return i;
    //         }
    //     }
    //     revert("Client not found");
    // }

    // function checkClientExist(uint _federatedCid) internal view returns (bool) {
    //     for (uint i = 0; i < clients.length; i++) {
    //         if (clients[i].federatedCid == _federatedCid) {
    //             return true;
    //         }
    //     }
    //     return false;
    // }

    function addToBlacklist(uint _federatedCid) public {
        blacklist.push(_federatedCid);
    }

    function getBlacklist() public view onlyOwner returns (uint[] memory) {
        return blacklist;
    }
}
