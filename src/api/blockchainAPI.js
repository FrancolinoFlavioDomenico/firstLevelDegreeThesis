import Fastify from "fastify";
import { ethers } from "ethers";
import { createRequire } from "module";
import { spawn } from 'node:child_process';
import * as fs from 'node:fs/promises';
import * as crypto from 'node:crypto';
const require = createRequire(import.meta.url);

const fastify = Fastify({
    logger: {
        enabled: true,
        file: "log/node_log.txt",
    },
    bodyLimit: 100 * 1024 * 1024,
});

fastify.register(require('@fastify/multipart'), {
    limits: {
        fileSize: 500 * 1024 * 1024,
    },
    attachFieldsToBody: "keyValues",
});

const provider = new ethers.JsonRpcProvider('http://127.0.0.1:8545/');

const compiledContract = require("../blockchain/artifacts/src/blockchain/contracts/CheckWeights.sol/CheckWeights.json");
const contractABI = compiledContract.abi;
const contractBytecode = compiledContract.bytecode;
var deployedContractAddress;

var datasetName = 'mnist'
var datasetClassNumber = 10
var maxRound = 5
var clientsNum = 10
function log(msg, type = "debug") {
    console.log(msg)
    fastify.log[type](msg);
}

function catchBlockchainEvent() {
    const contract = new ethers.Contract(deployedContractAddress, contractABI, provider);
    contract.on("WroteWeightsOfRoundForClient", (round, federatedCid, event) => {
        try {
            const child = spawn('python', ['src/utils/guard.py', datasetName, datasetClassNumber, maxRound, clientsNum, round, federatedCid], { shell: true });//run python script for weight check
            child.stderr.pipe(process.stdout)
        } catch (error) {
            log('Error executing Python script: ' + error, "error");
        }

    }
    )
};

/*********************************************************
 blockchain test route
 **********************************************************/

//----------connection test-----------
fastify.get("/isConnected", async function (request, reply) {
    const isConnected = await !!provider.getNetwork()
    reply.send({ isConnected: isConnected });
});

/*----------------------------------------------------------
                   real used api
----------------------------------------------------------*/
//configure federated train
fastify.post("/configure/training", async function (request, reply) {
    try {
        datasetName = request.body.datasetName;
        datasetClassNumber = request.body.datasetClassNumber;
        maxRound = request.body.maxRound;
        clientsNum = request.body.clientsNum;
        reply.send();
    } catch (e) {
        request.log.error({
            e,
            message: "deploy contract failed:" + e,
        });
        reply.code(500).send({ error: "Internal Server Error" });
    } finally {
        return reply
    }
});

//deploy smart contract on blockchain
fastify.post("/deploy/contract", async function (request, reply) {
    try {
        const wallet = new ethers.Wallet(request.body.blockchainCredential, provider)
        const contractFactory = new ethers.ContractFactory(contractABI, contractBytecode, wallet);
        const contract = await contractFactory.deploy();
        const deployTransaction = await contract.waitForDeployment();
        deployedContractAddress = deployTransaction.target

        reply.send({ deployTransaction: deployTransaction });
        catchBlockchainEvent();
    } catch (e) {
        request.log.error({
            e,
            message: "deploy contract failed:" + e,
        });
        reply.code(500).send({ error: "Internal Server Error" });
    } finally {
        return reply
    }
});


//write weights hash of client {:clientCid} at round {:round} into blockchain
fastify.post("/write/weights/:clientCid/:round", async (request, reply) => {

    try {
        const round = request.params.round;
        const clientCid = request.params.clientCid == 'server' ? clientsNum : request.params.clientCid;
        const weights = request.body.weights

        const walletKey = request.body.blockchainCredential
        const wallet = new ethers.Wallet(walletKey, provider)
        const contract = new ethers.Contract(deployedContractAddress, contractABI, wallet);

        const path = `./data/clientParameters/node/${clientCid == clientsNum ? 'server' : 'client' + clientCid}_round${round}_parameters.pth`
        await fs.writeFile(path, weights, { encoding: "binary" });
        const fileContent = await fs.readFile(path)

        const weightHash = crypto.createHash('md5').update(fileContent).digest('hex');

        const gastEstimated = await contract.addRoundWeightsReference.estimateGas(
            weightHash,
            round,
            clientCid
        );
        const tx = await contract.addRoundWeightsReference(
            weightHash,
            round,
            clientCid,
            { gasLimit: gastEstimated }
        )
        const txResult = await tx.wait()
        reply.send({ txResult: txResult, weightsHash: weightHash })
    } catch (e) {
        request.log.error({
            e,
            message: "write weight on blockchain failed:" + e,
        });
        reply.code(500).send({ error: "Internal Server Error" });

    } finally {
        return reply
    }
});

//get weights hash of client {:clientCid} at round {:round} into blockchain
fastify.get("/checksum/weights/:clientCid/:round", async (request, reply) => {

    try {
        const round = request.params.round;
        const clientCid = request.params.clientCid;

        const contract = new ethers.Contract(deployedContractAddress, contractABI, provider);
        const weightHash = await contract.getWeightOfRoundOfClient(round, clientCid)

        reply.send(weightHash)
    } catch (e) {
        request.log.error({
            e,
            message: "get weight hash on blockchain failed:" + e,
        });
        reply.code(500).send({ error: "Internal Server Error" });

    } finally {
        return reply
    }
});

//write poisoner client {:poisonerCid}  into blockchain blacklist
fastify.post("/write/blacklist/:poisonerCid", async (request, reply) => {
    try {
        const poisonerCid = request.params.poisonerCid;
        let walletKey = request.body.blockchainCredential
        const wallet = new ethers.Wallet(walletKey, provider)
        const contract = new ethers.Contract(deployedContractAddress, contractABI, wallet);

        const blacklist = await contract.isPoisonerCid(poisonerCid)

        if (!blacklist) {
            const gastEstimated = await contract.addToBlacklist.estimateGas(poisonerCid);
            const tx = await contract.addToBlacklist(
                poisonerCid,
                { gasLimit: gastEstimated }
            )
            const txResult = await tx.wait()

            reply.send({ txResult: txResult })
        } else {
            reply.send()
        }
    } catch (e) {
        request.log.error({
            e,
            message: "write blacklist on blockchain failed:" + e,
        });
        reply.code(500).send({ error: "Internal Server Error" });

    } finally {
        return reply
    }
});

//check if client is into blacklist
fastify.get("/poisoners/:clientCid", async (request, reply) => {
    try {
        const clientCid = request.params.clientCid;

        const contract = new ethers.Contract(deployedContractAddress, contractABI, provider);
        const blacklist = await contract.isPoisonerCid(clientCid)

        reply.send(blacklist)
    } catch (e) {
        request.log.error({
            e,
            message: "get blacklist failed:" + e,
        });
        reply.code(500).send({ error: "Internal Server Error" });

    } finally {
        return reply
    }
});


//check if server is corrupted
fastify.get("/server/isCorrupted", async (request, reply) => {
    try {
        const contract = new ethers.Contract(deployedContractAddress, contractABI, provider);
        const serverIsCorrupted = await contract.serverCorrupted(clientCid)

        reply.send(serverIsCorrupted);
    } catch (e) {
        request.log.error({
            e,
            message: "Internal Server Error:" + e,
        });
        reply.code(500).send({ error: "Internal Server Error" });

    } finally {
        return reply
    }
});


//write flag that server result corrupted
fastify.post("/server/corrupted/:isCorrupted", async (request, reply) => {

    try {
        const walletKey = request.body.blockchainCredential
        const wallet = new ethers.Wallet(walletKey, provider)
        const contract = new ethers.Contract(deployedContractAddress, contractABI, wallet);

        const tx = await contract.setServerCorrupted(request.params.isCorrupted == 'true')
        const txResult = tx.wait()

        reply.send(txResult)
    } catch (e) {
        request.log.error({
            e,
            message: "write flag on blockchain failed:" + e,
        });
        reply.code(500).send({ error: "Internal Server Error" });

    } finally {
        return reply
    }
});


/*----------------------------------------------------------
              start server
----------------------------------------------------------*/
fastify.listen({ port: 3000 }, function (err, address) {
    if (err) {
        fastify.log.error(err);
        process.exit(1);
    }
    log(`Server is now listening on ${address}`);
});